"""Assign triangles to zoning regions by centroid.

A zoning rule is a 3-tuple of monotone interval arrays. Each triangle's
centroid lands in exactly one (x-bin, y-bin, z-bin) box; that box is
its region id. Triangles whose centroid falls outside any axis's
declared intervals get ``-1`` (ungrouped).

The partitioning itself is delegated to the canonical triangle-grouping
pipeline in :mod:`cfdmod.geometry.grouping` (a :class:`ByZoningGrouping`
spec applied via :func:`apply_groupings`); this op only adapts the
result into the per-element index array carried by a v3 :class:`Grouping`.
Region ids are 0-indexed integers in raster order
(``ix + nx * iy + nx * ny * iz``).
"""

from __future__ import annotations

__all__ = ["ZoningGroupingParams", "zoning_grouping"]

import pathlib
from typing import ClassVar, Literal

import numpy as np
from lnas import LnasFormat
from pydantic import ConfigDict

from cfdmod.core.data_source import DataSource
from cfdmod.core.grouping import Grouping
from cfdmod.core.ops import OpParams
from cfdmod.geometry.grouping import ByZoningGrouping, apply_groupings


class ZoningGroupingParams(OpParams):
    """Parameters for :func:`zoning_grouping`.

    Attributes:
        mesh: Path to a ``.lnas`` file. Centroids are recomputed from the
            mesh so this op is independent of upstream :func:`mesh_attach`.
        x_intervals: Monotone increasing edges along x. ``n`` edges
            produce ``n - 1`` bins. Empty list -> no x-binning.
        y_intervals: Same convention along y.
        z_intervals: Same convention along z.
        name: Grouping name to attach. Defaults to ``"zone"``.
    """

    model_config = ConfigDict(frozen=True)

    kind: Literal["zoning_grouping"] = "zoning_grouping"
    mesh: str
    x_intervals: list[float] = []
    y_intervals: list[float] = []
    z_intervals: list[float] = []
    name: str = "zone"

    chunkable_along: ClassVar[frozenset[str]] = frozenset({"time"})


def _normalize_intervals(edges: list[float]) -> list[float]:
    """Empty -> open infinite interval; otherwise pass through."""
    if not edges:
        return [float("-inf"), float("inf")]
    return edges


def zoning_grouping(ds: DataSource, p: ZoningGroupingParams) -> DataSource:
    lnas = LnasFormat.from_file(pathlib.Path(p.mesh))
    if lnas.geometry.triangles.shape[0] != ds.n_elements:
        raise ValueError(
            f"mesh has {lnas.geometry.triangles.shape[0]} triangles but data source has "
            f"{ds.n_elements} elements."
        )

    # Raster region id uses (nx, ny); nz only informs name uniqueness via
    # the per-cell "ix-iy-iz" group names produced by apply_groupings.
    nx = max(len(p.x_intervals) - 1, 1)
    ny = max(len(p.y_intervals) - 1, 1)

    spec = ByZoningGrouping(
        x_intervals=_normalize_intervals(p.x_intervals),
        y_intervals=_normalize_intervals(p.y_intervals),
        z_intervals=_normalize_intervals(p.z_intervals),
        name_template="{ix}-{iy}-{iz}",
    )
    result = apply_groupings(lnas, [spec])

    indices = np.full(ds.n_elements, -1, dtype=np.int32)
    for name, tris in result.groups.items():
        ix_s, iy_s, iz_s = name.split("-")
        ix, iy, iz = int(ix_s), int(iy_s), int(iz_s)
        region_id = ix + nx * iy + (nx * ny) * iz
        indices[tris] = region_id

    return ds.with_grouping(Grouping(name=p.name, indices=indices))
