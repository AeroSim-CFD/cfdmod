"""Group triangles by a uniform NxNxN grid over the candidate bounding box.

Convenience wrapper over :class:`ByZoningGrouping` that builds equally
spaced bin edges from the bounding box of the candidate triangle
centroids (the candidate set is the parent mesh by default, or the
union of ``restrict_to`` groups when set).

Use this kind when the natural input is "divide each axis into N parts"
rather than explicit edge coordinates.
"""

from __future__ import annotations

from typing import Annotated, Literal

import numpy as np
from lnas import LnasFormat
from pydantic import BaseModel, Field

from cfdmod.geometry.grouping.kinds.by_zoning import ByZoningGrouping, apply_by_zoning


class ByDivisionsGrouping(BaseModel):
    """Cartesian binning by uniform division count per axis.

    Args:
        kind: Discriminator literal, always ``"by_divisions"``.
        n_div_x, n_div_y, n_div_z: Number of cells along each axis.
            ``None`` (the default) means "do not bin along this axis";
            the axis contributes a single cell spanning ``[-inf, inf]``.
        name_template: Format string for group names. Available
            placeholders: ``{idx}`` (linear region index, 0-based),
            ``{ix}``, ``{iy}``, ``{iz}`` (per-axis cell indices).
        restrict_to: Optional list of earlier group names; when set,
            only triangles in (the union of) those groups are considered
            and the bounding box is computed from their centroids only.
    """

    kind: Literal["by_divisions"] = "by_divisions"
    n_div_x: Annotated[
        int | None,
        Field(None, ge=1, description="Number of cells along x; None = no x binning."),
    ]
    n_div_y: Annotated[
        int | None,
        Field(None, ge=1, description="Number of cells along y; None = no y binning."),
    ]
    n_div_z: Annotated[
        int | None,
        Field(None, ge=1, description="Number of cells along z; None = no z binning."),
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


def _intervals_from_count(lo: float, hi: float, n: int | None) -> list[float]:
    """Build edges for ``n`` equal cells in ``[lo, hi]``, or no-bin sentinel.

    Edges are snapped to float32 precision (matching the precision of
    ``LnasGeometry`` vertex/centroid data) so a centroid that "should"
    land on an interior boundary is binned consistently rather than
    falling into the lower cell because of a float32->float64 round-up
    in the edge.
    """
    if n is None:
        return [float("-inf"), float("inf")]
    edges = np.linspace(lo, hi, n + 1).astype(np.float32).astype(np.float64)
    # Pad upper edge so the max-coordinate centroid (now exactly equal to
    # edges[-1] in float32) lands inside the last cell, not outside it.
    edges[-1] = np.nextafter(edges[-1], np.inf)
    return edges.tolist()


def apply_by_divisions(
    spec: ByDivisionsGrouping,
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
        x_intervals=_intervals_from_count(float(lo[0]), float(hi[0]), spec.n_div_x),
        y_intervals=_intervals_from_count(float(lo[1]), float(hi[1]), spec.n_div_y),
        z_intervals=_intervals_from_count(float(lo[2]), float(hi[2]), spec.n_div_z),
        name_template=spec.name_template,
    )
    return apply_by_zoning(inner, mesh, allowed)
