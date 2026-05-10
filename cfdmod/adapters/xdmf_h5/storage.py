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

import pathlib
from typing import Iterable

import h5py
import numpy as np

from cfdmod.adapters.xdmf_h5.field_store import H5FieldStore
from cfdmod.core.data_source import (
    DataSource,
    PointsDataSource,
    SurfaceDataSource,
)
from cfdmod.core.field_meta import FieldMeta
from cfdmod.core.time_axis import TimeAxis
from cfdmod.core.topology import ElementMeta, Topology
from cfdmod.io import xdmf as _xdmf

_RESERVED_ROOT_KEYS = frozenset({"Triangles", "Geometry", "Connectivity", "meta"})


def _kind_from_key(key: str) -> str:
    """Infer DataSource kind from the filename stem.

    The cfdmod v2 file-naming convention prefixes each h5 with the
    target kind: ``bodies.*``, ``cp_t.*``, ``stats.*`` for surfaces;
    ``points.*`` for probes. Anything else defaults to surface, which
    is the most common case.
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
            raise KeyError(f"XdmfH5Storage has no data source under key {key!r} ({h5_path})")

        with h5py.File(h5_path, "r") as f:
            if "Triangles" not in f or "Geometry" not in f:
                raise ValueError(
                    f"{h5_path} is missing the standard /Triangles and /Geometry datasets; "
                    "this adapter only handles the cfdmod v2 layout."
                )
            triangles = np.asarray(f["Triangles"][:], dtype=np.int32)
            vertices = np.asarray(f["Geometry"][:], dtype=np.float64)
            has_meta = "meta" in f and "time_steps" in f["meta"]
            time_steps = np.asarray(f["meta"]["time_steps"][:], dtype=np.float64) if has_meta else None
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
                children = [k for k in obj.keys() if isinstance(obj[k], h5py.Dataset)]
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
                        field_groups[f"{name}/{stat_name}"] = f"{name}/{stat_name}"

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
                    group = "stats"
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
    raise ValueError(
        f"XdmfH5Storage cannot write topology with cell_type={topology.cell_type!r}"
    )


def _is_floatish(s: str) -> bool:
    try:
        float(s)
    except ValueError:
        return False
    return True
