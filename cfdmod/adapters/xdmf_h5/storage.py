"""XDMF + H5 :class:`Storage`.

Reads and writes :class:`DataSource` objects to disk in the same byte
layout the v2 pressure pipeline produces today. Round-trip is the
contract: read a fixture, write it under a new key, the bytes match.

Layouts handled:

- *Timeseries* (the common case)::

      /Triangles                int32   (n_triangles, 3)
      /Geometry                 float64 (n_vertices, 3)
      /meta/time_steps          float64 (n_timesteps,)
      /meta/time_normalized     float64 (n_timesteps,)
      /{field}/t{T}             float64 (n_elements,)        per timestep

- *Time-aggregated* (stats)::

      /Triangles, /Geometry as above
      /{group}/{stat_name}      float64 (n_elements,)

The file at ``<root>/<key>.h5`` together with the optional sidecar
``<root>/<key>.xdmf`` is the storage unit. ``read_data_source`` returns
a :class:`SurfaceDataSource` or :class:`PointsDataSource`; the kind is
inferred from the ``key`` filename prefix (``bodies.``/``cp_t.``/...
-> surface, ``points.`` -> points). ``write_data_source`` rewrites the
file from scratch using the existing ``cfdmod.io.xdmf`` helpers, so the
output format is exactly what the v2 pipeline produces.
"""

from __future__ import annotations

__all__ = ["XdmfH5Storage"]

import hashlib
import pathlib
from typing import Iterable

import h5py
import numpy as np

from cfdmod.adapters.xdmf_h5.field_store import H5FieldStore
from cfdmod.core.data_source import (
    DataSource,
    GroupsDataSource,
    PointsDataSource,
    SurfaceDataSource,
)
from cfdmod.core.errors import StorageKeyError
from cfdmod.core.field_meta import FieldMeta
from cfdmod.core.time_axis import TimeAxis
from cfdmod.core.topology import ElementMeta, Topology
from cfdmod.io import xdmf as _xdmf

# Root-level h5 attribute the freshness layer stamps an output's signature
# under. It lives in ``.attrs`` (not a dataset), so ``read_data_source``
# ignores it and the round-trip byte layout is unchanged.
_SIGNATURE_ATTR = "cfdmod_signature"

_RESERVED_ROOT_KEYS = frozenset({"Triangles", "Geometry", "Connectivity", "meta"})
# Geometry datasets embedded inside a stats group (so write_stats_xdmf can
# emit one Grid per group). They are topology, not stat fields, and must be
# skipped when reconstructing the field list on read.
_RESERVED_GROUP_DATASETS = frozenset({"Triangles", "Geometry", "Connectivity"})
# Synthetic h5 group holding stats that had no group prefix on the source
# (e.g. a Cp stats source with bare fields mean/rms/...). Stripped on read so
# the round-trip restores the original bare field names.
_BARE_STATS_GROUP = "stats"


def _kind_from_key(key: str) -> str:
    """Infer DataSource kind from the filename stem.

    A ``points.*`` stem is read as a points (probe) source; every other
    stem defaults to surface, which is the most common case (``bodies.*``,
    ``cp_t.*``, ``stats.*``, ...). The prefix is the only signal -- a probe
    file must therefore be named ``points.*`` to be read back as points.
    """
    stem = pathlib.Path(key).name
    if stem.startswith("points."):
        return "points"
    return "surface"


def _derive_time_axis(time_steps: np.ndarray, time_normalized: np.ndarray) -> TimeAxis:
    """Reconstruct an affine :class:`TimeAxis` from the stored arrays.

    The on-disk arrays are kept as-is; we only read them to pull the
    three numbers (initial_time, timestep_size, n_timesteps) plus the
    normalization offset. If the on-disk arrays are not strictly
    uniform, we still take ``timestep_size`` from the first delta --
    this matches what the v2 pipeline produces (regular sampling) and
    avoids the cost of resampling on read.
    """
    n = int(time_steps.shape[0])
    if n == 0:
        return TimeAxis(initial_time=0.0, timestep_size=0.0, n_timesteps=0)
    if n == 1:
        return TimeAxis(
            initial_time=float(time_steps[0]),
            timestep_size=1.0,
            n_timesteps=1,
            time_normalized_offset=float(time_steps[0] - time_normalized[0]),
        )
    dt = float(time_steps[1] - time_steps[0])
    offset = float(time_steps[0] - time_normalized[0])
    return TimeAxis(
        initial_time=float(time_steps[0]),
        timestep_size=dt,
        n_timesteps=n,
        time_normalized_offset=offset,
    )


class XdmfH5Storage:
    """:class:`Storage` for the XDMF + H5 byte layout.

    Args:
        root: Directory under which keys resolve. ``read_data_source("bodies.foo")``
            opens ``<root>/bodies.foo.h5``.
        write_xdmf: When True, ``write_data_source`` also (re)generates
            ``<key>.xdmf`` next to the h5. Default True.
    """

    __slots__ = ("_root", "_write_xdmf")

    def __init__(self, root: pathlib.Path, *, write_xdmf: bool = True) -> None:
        self._root = pathlib.Path(root)
        self._write_xdmf = bool(write_xdmf)

    # --- Path helpers ------------------------------------------------------

    @property
    def root(self) -> pathlib.Path:
        return self._root

    def h5_path(self, key: str) -> pathlib.Path:
        return self._root / f"{key}.h5"

    def xdmf_path(self, key: str) -> pathlib.Path:
        return self._root / f"{key}.xdmf"

    def keys(self) -> Iterable[str]:
        if not self._root.exists():
            return []
        return sorted(p.stem for p in self._root.glob("*.h5"))

    def __contains__(self, key: str) -> bool:
        return self.h5_path(key).exists()

    # --- Read --------------------------------------------------------------

    def read_data_source(self, key: str) -> DataSource:
        h5_path = self.h5_path(key)
        if not h5_path.exists():
            raise StorageKeyError(
                f"XdmfH5Storage has no data source under key {key!r} ({h5_path})"
            )

        with h5py.File(h5_path, "r") as f:
            if "Triangles" not in f or "Geometry" not in f:
                raise ValueError(
                    f"{h5_path} is missing the standard /Triangles and /Geometry datasets; "
                    "this adapter only handles the cfdmod v2 layout."
                )
            triangles = np.asarray(f["Triangles"][:], dtype=np.int32)
            vertices = np.asarray(f["Geometry"][:], dtype=np.float64)
            has_meta = "meta" in f and "time_steps" in f["meta"]
            time_steps = (
                np.asarray(f["meta"]["time_steps"][:], dtype=np.float64) if has_meta else None
            )
            time_normalized = (
                np.asarray(f["meta"]["time_normalized"][:], dtype=np.float64) if has_meta else None
            )

            field_groups: dict[str, str] = {}
            time_keys: list[str] = []
            time_aggregated = False

            # Field groups are top-level groups other than 'meta'. Detect
            # timeseries vs stats by inspecting one group's children.
            for name in f.keys():
                if name in _RESERVED_ROOT_KEYS:
                    continue
                obj = f[name]
                if not isinstance(obj, h5py.Group):
                    continue
                children = [
                    k
                    for k in obj.keys()
                    if isinstance(obj[k], h5py.Dataset) and k not in _RESERVED_GROUP_DATASETS
                ]
                if not children:
                    continue
                # Stats layout: per-stat datasets directly under the group, no t-prefix.
                # Timeseries layout: every dataset is t{float}.
                if all(k.startswith("t") and _is_floatish(k[1:]) for k in children):
                    field_groups[name] = name
                    if not time_keys:
                        time_keys = sorted(children, key=lambda k: float(k[1:]))
                else:
                    time_aggregated = True
                    for stat_name in children:
                        # Bare-stat sources are written under the synthetic
                        # "stats" group; strip it so the field name round-trips.
                        field_name = (
                            stat_name if name == _BARE_STATS_GROUP else f"{name}/{stat_name}"
                        )
                        field_groups[field_name] = f"{name}/{stat_name}"

        # Topology + ElementMeta
        kind = _kind_from_key(key)
        if kind == "points":
            topology = Topology.points(vertices)
        else:
            topology = Topology.triangles(triangles, vertices)
        elements = ElementMeta()

        # Time axis
        if time_aggregated or not time_keys:
            if has_meta and time_steps.shape[0] > 0 and not time_aggregated:
                time = _derive_time_axis(time_steps, time_normalized)
            else:
                time = TimeAxis(initial_time=0.0, timestep_size=0.0, n_timesteps=0)
        else:
            if has_meta:
                time = _derive_time_axis(time_steps, time_normalized)
            else:
                # Reconstruct from the keys themselves.
                ts = np.array([float(k[1:]) for k in time_keys], dtype=np.float64)
                time = _derive_time_axis(ts, ts)

        store = H5FieldStore(
            h5_path=h5_path,
            field_groups=field_groups,
            time_keys=[] if time_aggregated else time_keys,
            n_elements=topology.n_elements,
            time_aggregated=time_aggregated,
        )
        field_meta = {name: FieldMeta(name=name) for name in field_groups}

        common = dict(
            time=time,
            topology=topology,
            elements=elements,
            fields=store,
            field_meta=field_meta,
            attrs={"source_path": str(h5_path)},
        )
        if kind == "points":
            return PointsDataSource(**common)
        return SurfaceDataSource(**common)

    # --- Write -------------------------------------------------------------

    def write_data_source(self, key: str, ds: DataSource) -> None:
        # GroupsDataSource is special: it has no topology of its own,
        # but it does carry parent_topology + parent_grouping. We
        # broadcast per-group values back to the parent's triangles so
        # the on-disk h5 is a regular surface timeseries that ParaView
        # can render. The result matches the legacy run_cf output.
        if isinstance(ds, GroupsDataSource):
            ds = _groups_to_parent_surface(ds)

        if ds.topology is None:
            raise ValueError(
                f"XdmfH5Storage cannot write a DataSource with no topology (kind={ds.kind!r})."
            )
        h5_path = self.h5_path(key)
        h5_path.parent.mkdir(parents=True, exist_ok=True)
        if h5_path.exists():
            h5_path.unlink()

        triangles = _connectivity_for_write(ds.topology)
        vertices = np.asarray(ds.topology.vertices, dtype=np.float64)
        _xdmf.write_timeseries_geometry(h5_path, triangles, vertices)

        time_aggregated = ds.time.is_time_aggregated
        if not time_aggregated:
            time_steps = ds.time.times()
            time_normalized = ds.time.times_normalized()
            _xdmf.write_timeseries_meta(h5_path, time_steps, time_normalized)

        # Resolve every field by reading via the source's own FieldStore.
        # That makes the writeback work uniformly for MemoryFieldStore,
        # H5FieldStore (with overlay), and any future store.
        groups_for_xdmf: list[str] = []
        for fname in sorted(ds.fields.keys()):
            arr = ds.fields.read(fname)
            if time_aggregated:
                # Single dataset path: support 'group/stat' or bare 'stat'.
                group, _, stat = fname.partition("/")
                if not stat:
                    stat = group
                    group = _BARE_STATS_GROUP
                _xdmf.write_stats_field(
                    h5_path,
                    group=group,
                    stat_name=stat,
                    values=np.asarray(arr, dtype=np.float64),
                    triangles=triangles,
                    vertices=vertices,
                )
            else:
                # Timeseries: arr is (n_elements, n_timesteps).
                if arr.ndim != 2:
                    raise ValueError(
                        f"field {fname!r} must be 2-D for a non-aggregated DataSource; "
                        f"got shape {arr.shape}"
                    )
                ts = ds.time.times()
                for i, t in enumerate(ts):
                    _xdmf.write_timeseries_step(
                        h5_path,
                        group=fname,
                        key=f"t{t}",
                        data=np.asarray(arr[:, i], dtype=np.float64),
                    )
                groups_for_xdmf.append(fname)

        if self._write_xdmf:
            xdmf_path = self.xdmf_path(key)
            if time_aggregated:
                _xdmf.write_stats_xdmf(h5_path, xdmf_path)
            elif groups_for_xdmf:
                _xdmf.write_temporal_xdmf(h5_path, xdmf_path, groups_for_xdmf)

    # --- Freshness ---------------------------------------------------------

    def digest(self, key: str, strategy: str = "size_mtime") -> str:
        """Change-detecting token for the ``<key>.h5`` (+ ``.xdmf``) pair.

        - ``size_mtime`` (default): size + mtime of the file(s); no reads.
        - ``content``: a blake2b hash of the file bytes, streamed.
        - ``backend``: the local filesystem has no native token, so this
          falls back to ``size_mtime`` (tagged so the fallback is visible).
        """
        h5_path = self.h5_path(key)
        if not h5_path.exists():
            raise StorageKeyError(
                f"XdmfH5Storage has no data source under key {key!r} ({h5_path})"
            )
        paths = [h5_path]
        xdmf = self.xdmf_path(key)
        if xdmf.exists():
            paths.append(xdmf)

        if strategy == "content":
            h = hashlib.blake2b(digest_size=32)
            for p in paths:
                with open(p, "rb") as f:
                    for block in iter(lambda: f.read(1 << 20), b""):
                        h.update(block)
            return f"content:{h.hexdigest()}"

        # size_mtime and backend (backend has no FS-native token -> fall back)
        prefix = "size_mtime" if strategy != "backend" else "backend_fs"
        parts = [f"{p.stat().st_size}:{p.stat().st_mtime_ns}" for p in paths]
        return f"{prefix}:" + "|".join(parts)

    def read_signature(self, key: str) -> str | None:
        h5_path = self.h5_path(key)
        if not h5_path.exists():
            return None
        with h5py.File(h5_path, "r") as f:
            raw = f.attrs.get(_SIGNATURE_ATTR)
        if raw is None:
            return None
        return raw.decode("utf-8") if isinstance(raw, bytes) else str(raw)

    def write_signature(self, key: str, signature: str) -> None:
        h5_path = self.h5_path(key)
        if not h5_path.exists():
            raise StorageKeyError(f"cannot stamp signature: no h5 under key {key!r} ({h5_path})")
        with h5py.File(h5_path, "a") as f:
            f.attrs[_SIGNATURE_ATTR] = signature


def _groups_to_parent_surface(ds: GroupsDataSource) -> SurfaceDataSource:
    """Broadcast a GroupsDataSource back onto its parent surface.

    Returns a :class:`SurfaceDataSource` over the parent's triangles
    with each parent triangle taking the value of the group it belongs
    to. Triangles in ungrouped territory (``-1``) get NaN.
    """
    from cfdmod.adapters.memory import MemoryFieldStore
    from cfdmod.core.field_meta import FieldMeta

    parent_indices = ds.parent_grouping.indices
    group_ids = ds.groupings[ds.parent_grouping.name].indices  # row index -> group id
    # Map group id -> row index in the groups source.
    row_for_gid = {int(gid): row for row, gid in enumerate(group_ids)}
    n_parent = ds.parent_topology.n_elements

    out_arrays: dict[str, np.ndarray] = {}
    out_meta: dict[str, FieldMeta] = {}
    for fname in ds.fields.keys():
        arr = np.asarray(ds.fields.read(fname), dtype=np.float64)
        if arr.ndim == 2:
            broadcast = np.full((n_parent, arr.shape[1]), np.nan, dtype=np.float64)
        else:
            broadcast = np.full(n_parent, np.nan, dtype=np.float64)
        for tri in range(n_parent):
            gid = int(parent_indices[tri])
            if gid not in row_for_gid:
                continue
            broadcast[tri] = arr[row_for_gid[gid]]
        out_arrays[fname] = broadcast
        out_meta[fname] = ds.field_meta.get(fname, FieldMeta(name=fname))

    return SurfaceDataSource(
        time=ds.time,
        topology=ds.parent_topology,
        elements=ElementMeta(),
        fields=MemoryFieldStore(out_arrays),
        field_meta=out_meta,
    )


def _connectivity_for_write(topology: Topology) -> np.ndarray:
    """Connectivity array as written under ``/Triangles``.

    Triangle topologies write their connectivity directly. Point
    topologies write a degenerate ``(n_points, 3)`` block of point
    indices so the on-disk layout still has a ``/Triangles`` dataset --
    this matches the v2 ``points.*.h5`` files in the fixtures.
    """
    if topology.cell_type == "triangle":
        return np.asarray(topology.connectivity, dtype=np.int32)
    if topology.cell_type == "point":
        n = topology.n_elements
        col = np.arange(n, dtype=np.int32)
        return np.stack([col, col, col], axis=1)
    raise ValueError(f"XdmfH5Storage cannot write topology with cell_type={topology.cell_type!r}")


def _is_floatish(s: str) -> bool:
    try:
        float(s)
    except ValueError:
        return False
    return True
