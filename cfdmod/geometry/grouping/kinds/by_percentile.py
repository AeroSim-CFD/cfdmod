"""Quantile binning along one axis: each bucket holds ~equal triangle count.

For each candidate triangle, take its centroid coordinate along ``axis``;
compute ``n_quantiles`` equal-count bins via the empirical quantiles of
that 1D distribution; assign the triangle to its bucket. Useful when
triangle density along the axis varies, so equal-width cells (as in
:class:`ByDivisionsGrouping`) would produce wildly uneven counts.
"""

from __future__ import annotations

from typing import Annotated, Literal

import numpy as np
from lnas import LnasFormat
from pydantic import BaseModel, Field

_AXIS_INDEX = {"x": 0, "y": 1, "z": 2}


class ByPercentileGrouping(BaseModel):
    """Equal-count quantile binning along one axis.

    Args:
        kind: Discriminator literal, always ``"by_percentile"``.
        axis: Which axis to bin along: ``"x"``, ``"y"``, or ``"z"``.
        n_quantiles: Number of equal-count bins (>= 1).
        name_template: Format string. Placeholder ``{idx}`` (0-based).
        restrict_to: Optional list of earlier group names to scope to.
    """

    kind: Literal["by_percentile"] = "by_percentile"
    axis: Annotated[
        Literal["x", "y", "z"],
        Field(description="Axis to bin along."),
    ]
    n_quantiles: Annotated[
        int,
        Field(ge=1, description="Number of equal-count bins."),
    ]
    name_template: Annotated[
        str,
        Field("q{idx}", description="Format string; placeholder: {idx}."),
    ]
    restrict_to: Annotated[
        list[str] | None,
        Field(None, description="Optional list of earlier group names to restrict to."),
    ]


def apply_by_percentile(
    spec: ByPercentileGrouping,
    mesh: LnasFormat,
    allowed: np.ndarray | None,
) -> dict[str, np.ndarray]:
    """Bin candidate centroids by empirical quantiles along the chosen axis."""
    triangles = mesh.geometry.triangle_vertices
    centroids = np.mean(triangles, axis=1)
    n_parent = centroids.shape[0]

    if allowed is not None:
        cand = np.asarray(allowed, dtype=np.int64)
    else:
        cand = np.arange(n_parent, dtype=np.int64)

    if cand.size == 0:
        return {}

    coords = centroids[cand, _AXIS_INDEX[spec.axis]].astype(np.float64)

    n = spec.n_quantiles
    if n == 1:
        return {spec.name_template.format(idx=0): np.sort(cand)}

    edges = np.quantile(coords, np.linspace(0.0, 1.0, n + 1))
    # Pad upper edge so the max-coordinate centroid lands in the last cell.
    edges[-1] = np.nextafter(edges[-1], np.inf)

    out: dict[str, np.ndarray] = {}
    for i in range(n):
        lo, hi = edges[i], edges[i + 1]
        in_cell = (coords >= lo) & (coords < hi)
        cell_idxs = cand[in_cell]
        if cell_idxs.size == 0:
            continue
        name = spec.name_template.format(idx=i)
        if name in out:
            raise ValueError(
                f"ByPercentileGrouping: name_template {spec.name_template!r} produced "
                f"duplicate group name {name!r}; include {{idx}} for uniqueness"
            )
        out[name] = cell_idxs
    return out
