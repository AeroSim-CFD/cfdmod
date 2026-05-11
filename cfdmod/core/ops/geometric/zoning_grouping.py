"""Assign triangles to zoning regions by centroid.

A zoning rule is a 3-tuple of monotone interval arrays. Each triangle's
centroid lands in exactly one (x-bin, y-bin, z-bin) box; that box is
its region id. Triangles whose centroid falls outside any axis's
declared intervals get ``-1`` (ungrouped).

Region ids are 0-indexed integers in raster order::

    region_id = i_x + n_x * i_y + n_x * n_y * i_z

where ``n_x = len(x_intervals) - 1`` etc.

The op assumes ``elements.position`` is populated (attach with
:func:`mesh_attach` first).

Out of scope here: triangle cutting along zone boundaries (the legacy
``process_surfaces`` machinery). Centroid-based assignment is the
common case and matches what cfdmod's Ce statistics actually compute.
"""

from __future__ import annotations

__all__ = ["ZoningGroupingParams", "zoning_grouping"]

from typing import ClassVar, Literal

import numpy as np
from pydantic import ConfigDict

from cfdmod.core.data_source import DataSource
from cfdmod.core.grouping import Grouping
from cfdmod.core.ops import OpParams


class ZoningGroupingParams(OpParams):
    """Parameters for :func:`zoning_grouping`.

    Attributes:
        x_intervals: Monotone increasing edges along x. ``n`` edges
            produce ``n - 1`` bins. Empty list -> no x-binning (every
            centroid passes).
        y_intervals: Same convention along y.
        z_intervals: Same convention along z.
        name: Grouping name to attach. Defaults to ``"zone"``.
    """

    model_config = ConfigDict(frozen=True)

    kind: Literal["zoning_grouping"] = "zoning_grouping"
    x_intervals: list[float] = []
    y_intervals: list[float] = []
    z_intervals: list[float] = []
    name: str = "zone"

    chunkable_along: ClassVar[frozenset[str]] = frozenset({"time"})


def _bin_along(values: np.ndarray, edges: list[float]) -> np.ndarray:
    """``-1`` for out-of-range, else 0-based bin index."""
    if not edges:
        return np.zeros(values.shape[0], dtype=np.int32)
    arr = np.asarray(edges, dtype=np.float64)
    if not np.all(np.diff(arr) > 0):
        raise ValueError(f"intervals must be strictly increasing; got {edges}")
    # np.searchsorted returns 1..n for values inside [edges[0], edges[-1]).
    idx = np.searchsorted(arr, values, side="right") - 1
    out = idx.astype(np.int32)
    in_range = (values >= arr[0]) & (values < arr[-1])
    out[~in_range] = -1
    return out


def zoning_grouping(ds: DataSource, p: ZoningGroupingParams) -> DataSource:
    if ds.elements.position is None:
        raise ValueError(
            "zoning_grouping requires elements.position; "
            "attach centroids with mesh_attach first."
        )

    centroids = ds.elements.position
    bx = _bin_along(centroids[:, 0], p.x_intervals)
    by = _bin_along(centroids[:, 1], p.y_intervals)
    bz = _bin_along(centroids[:, 2], p.z_intervals)

    nx = max(len(p.x_intervals) - 1, 1)
    ny = max(len(p.y_intervals) - 1, 1)

    region = bx + nx * by + (nx * ny) * bz
    invalid = (bx < 0) | (by < 0) | (bz < 0)
    region[invalid] = -1

    return ds.with_grouping(Grouping(name=p.name, indices=region))
