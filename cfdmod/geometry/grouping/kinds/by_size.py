"""Group triangles by a fixed-size grid over the candidate bounding box.

Convenience wrapper over :class:`ByZoningGrouping` that builds bin edges
of a user-specified cell size per axis, anchored at the lower corner of
the candidate bounding box. The number of cells along each axis is
``ceil((hi - lo) / size)``; the last cell may extend slightly past ``hi``
so the max-coordinate centroid is included.

Use this kind when the natural input is "cells of size S along each
axis" rather than a count of divisions.
"""

from __future__ import annotations

import math
from typing import Annotated, Literal

import numpy as np
from lnas import LnasFormat
from pydantic import BaseModel, Field

from cfdmod.geometry.grouping.kinds.by_zoning import ByZoningGrouping, apply_by_zoning


class BySizeGrouping(BaseModel):
    """Cartesian binning by fixed cell size per axis.

    Args:
        kind: Discriminator literal, always ``"by_size"``.
        size_x, size_y, size_z: Cell size along each axis (must be > 0).
            ``None`` (the default) means "do not bin along this axis";
            the axis contributes a single cell spanning ``[-inf, inf]``.
        name_template: Format string for group names. Available
            placeholders: ``{idx}`` (linear region index, 0-based),
            ``{ix}``, ``{iy}``, ``{iz}`` (per-axis cell indices).
        restrict_to: Optional list of earlier group names; when set,
            only triangles in (the union of) those groups are considered
            and the bounding box is computed from their centroids only.
    """

    kind: Literal["by_size"] = "by_size"
    size_x: Annotated[
        float | None,
        Field(None, gt=0.0, description="Cell size along x; None = no x binning."),
    ]
    size_y: Annotated[
        float | None,
        Field(None, gt=0.0, description="Cell size along y; None = no y binning."),
    ]
    size_z: Annotated[
        float | None,
        Field(None, gt=0.0, description="Cell size along z; None = no z binning."),
    ]
    name_template: Annotated[
        str,
        Field(
            "r{idx}",
            description=(
                "Format string for group names. Placeholders: "
                "{idx} (linear), {ix}, {iy}, {iz} (per-axis)."
            ),
        ),
    ]
    restrict_to: Annotated[
        list[str] | None,
        Field(
            None,
            description="Optional list of earlier group names to restrict binning to.",
        ),
    ]


def _intervals_from_size(lo: float, hi: float, size: float | None) -> list[float]:
    """Build edges for cells of ``size`` covering ``[lo, hi]``, or no-bin sentinel.

    Edges are snapped to float32 precision (matching ``LnasGeometry``
    vertex/centroid precision) so a centroid that "should" land on an
    interior boundary is binned consistently.
    """
    if size is None:
        return [float("-inf"), float("inf")]
    extent = hi - lo
    n_cells = max(1, math.ceil(extent / size)) if extent > 0 else 1
    edges_arr = np.array([lo + i * size for i in range(n_cells + 1)], dtype=np.float64)
    edges_arr = edges_arr.astype(np.float32).astype(np.float64)
    # Make the upper edge strictly exceed the max centroid so it is
    # included; ByZoning uses [lower, upper) semantics.
    hi_f32 = float(np.float32(hi))
    if edges_arr[-1] <= hi_f32:
        edges_arr[-1] = np.nextafter(hi_f32, np.inf)
    return edges_arr.tolist()


def apply_by_size(
    spec: BySizeGrouping,
    mesh: LnasFormat,
    allowed: np.ndarray | None,
) -> dict[str, np.ndarray]:
    """Compute the bbox-derived edges and delegate to ``apply_by_zoning``.

    Args:
        spec: The grouping spec.
        mesh: Parent mesh; uses ``mesh.geometry.triangle_vertices``.
        allowed: Optional sorted parent-triangle indices to restrict to.

    Returns:
        ``dict[group_name, sorted int64 parent triangle indices]``. Empty
        cells are omitted.
    """
    triangles = mesh.geometry.triangle_vertices  # (n_tri, 3, 3)
    centroids = np.mean(triangles, axis=1)  # (n_tri, 3)

    if allowed is not None:
        cand = np.asarray(allowed, dtype=np.int64)
    else:
        cand = np.arange(centroids.shape[0], dtype=np.int64)

    if cand.size == 0:
        return {}

    cand_centroids = centroids[cand]
    lo = cand_centroids.min(axis=0)
    hi = cand_centroids.max(axis=0)

    inner = ByZoningGrouping(
        x_intervals=_intervals_from_size(float(lo[0]), float(hi[0]), spec.size_x),
        y_intervals=_intervals_from_size(float(lo[1]), float(hi[1]), spec.size_y),
        z_intervals=_intervals_from_size(float(lo[2]), float(hi[2]), spec.size_z),
        name_template=spec.name_template,
    )
    return apply_by_zoning(inner, mesh, allowed)
