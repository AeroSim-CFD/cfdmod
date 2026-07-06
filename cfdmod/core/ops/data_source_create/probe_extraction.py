"""Probe extraction -- pull a field at a list of probe positions.

Given a source data source whose elements are positioned in 3D
(typically a :class:`VolumeDataSource` or :class:`PointsDataSource`)
and a list of probe positions, return a new
:class:`PointsDataSource` with the source's field interpolated /
nearest-sampled at each probe.

Two modes:

- ``"nearest"`` (default) -- map every probe to the closest source
  element. O(n_probes * n_source_elements). Suitable for moderate-size
  meshes.
- ``"linear_zaxis"`` -- 1-D linear interpolation along the z axis.
  Used by the S1 recipe to lift a CFD probe column onto a target
  height vector. Source elements must be sorted by z and lie on a
  single column (same x, y).
"""

from __future__ import annotations

__all__ = ["ProbeExtractionParams", "probe_extraction"]

from typing import Any, ClassVar, Literal

import numpy as np
from pydantic import ConfigDict

from cfdmod.adapters.memory import MemoryFieldStore
from cfdmod.core.data_source import DataSource, PointsDataSource
from cfdmod.core.field_meta import FieldMeta
from cfdmod.core.ops import OpParams
from cfdmod.core.topology import ElementMeta, Topology


class ProbeExtractionParams(OpParams):
    """Parameters for :func:`probe_extraction`.

    Attributes:
        probes: ``(n_probes, 3)`` probe positions. ``("linear_zaxis"``
            mode reads only the z column).
        field: Source field name.
        out: Output field name on the resulting points data source.
            Defaults to ``field``.
        mode: Extraction mode (``"nearest"`` or ``"linear_zaxis"``).
    """

    model_config = ConfigDict(frozen=True, arbitrary_types_allowed=True)

    kind: Literal["probe_extraction"] = "probe_extraction"
    probes: Any
    field: str
    out: str | None = None
    mode: Literal["nearest", "linear_zaxis"] = "nearest"

    chunkable_along: ClassVar[frozenset[str]] = frozenset({"time"})


def _nearest_indices(src_pos: np.ndarray, probes: np.ndarray) -> np.ndarray:
    out = np.empty(probes.shape[0], dtype=np.int64)
    for i in range(probes.shape[0]):
        d = ((src_pos - probes[i]) ** 2).sum(axis=1)
        out[i] = int(np.argmin(d))
    return out


def probe_extraction(ds: DataSource, p: ProbeExtractionParams) -> PointsDataSource:
    if ds.elements.position is None:
        raise ValueError("probe_extraction requires elements.position on the source")
    src_pos = ds.elements.position
    probes = np.asarray(p.probes, dtype=np.float64)
    if probes.ndim != 2 or probes.shape[1] != 3:
        raise ValueError(f"probes must have shape (n_probes, 3); got {probes.shape}")

    arr = np.asarray(ds.fields.read(p.field), dtype=np.float64)
    is_time = arr.ndim == 2

    if p.mode == "nearest":
        idx = _nearest_indices(src_pos, probes)
        sub = arr[idx]
    elif p.mode == "linear_zaxis":
        order = np.argsort(src_pos[:, 2])
        z_sorted = src_pos[order, 2]
        target_z = probes[:, 2]
        if is_time:
            n_probes = probes.shape[0]
            n_t = arr.shape[1]
            sub = np.empty((n_probes, n_t), dtype=np.float64)
            for t in range(n_t):
                sub[:, t] = np.interp(target_z, z_sorted, arr[order, t])
        else:
            sub = np.interp(target_z, z_sorted, arr[order])
    else:  # pragma: no cover -- guarded by Literal
        raise ValueError(f"unknown probe extraction mode {p.mode!r}")

    target = p.out or p.field
    src_meta = ds.field_meta.get(p.field)
    out_meta = (
        FieldMeta(name=target, unit=src_meta.unit, scale=src_meta.scale)
        if src_meta is not None
        else FieldMeta(name=target)
    )

    return PointsDataSource(
        time=ds.time,
        topology=Topology.points(probes),
        elements=ElementMeta(position=probes),
        fields=MemoryFieldStore({target: sub}),
        field_meta={target: out_meta},
    )
