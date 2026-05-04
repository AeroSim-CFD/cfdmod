"""Group triangle centroids by (r, theta, axial) bins around a cylinder axis.

For each candidate centroid ``c``, compute the cylindrical coordinates
relative to ``origin`` and ``axis``:

- ``r``     -- radial distance in the plane perpendicular to ``axis``.
- ``theta`` -- in-plane angle in degrees, normalised to ``[0, 360)``.
  Convention is right-handed cyclic: for ``axis="z"``, theta is measured
  from ``+x`` toward ``+y``; for ``axis="x"``, from ``+y`` toward ``+z``;
  for ``axis="y"``, from ``+z`` toward ``+x``.
- ``axial`` -- signed coordinate along ``axis``.

Each axis interval list (``r_intervals``, ``theta_intervals_deg``,
``axial_intervals``) declares bin edges; an axis with ``None`` is a
single cell. Cells whose Cartesian product would be empty are omitted.

Theta intervals do not wrap around; to express e.g. a sector spanning
``[350, 10]`` deg, supply two separate intervals (``[0, 10]`` and
``[350, 360]``) with a non-colliding template.
"""

from __future__ import annotations

import itertools
from typing import Annotated, Literal

import numpy as np
from lnas import LnasFormat
from pydantic import BaseModel, Field, field_validator

_AXIS_INDEX = {"x": 0, "y": 1, "z": 2}
_INPLANE: dict[str, tuple[int, int]] = {
    "x": (1, 2),  # theta from +y toward +z
    "y": (2, 0),  # theta from +z toward +x
    "z": (0, 1),  # theta from +x toward +y
}


class ByCylindricalGrouping(BaseModel):
    """Cartesian product of (r, theta, axial) bins around a cylinder axis.

    Args:
        kind: Discriminator literal, always ``"by_cylindrical"``.
        origin: Point on the cylinder axis (3-vector).
        axis: Cylinder axis: ``"x"``, ``"y"``, or ``"z"``.
        r_intervals: Radial bin edges (>= 0, ascending), or None.
        theta_intervals_deg: Angular bin edges in degrees, in [0, 360],
            ascending, or None.
        axial_intervals: Bin edges along ``axis`` (ascending), or None.
        name_template: Format string. Placeholders: ``{idx}`` (linear),
            ``{ir}``, ``{it}``, ``{iz}`` (per-axis cell indices).
        restrict_to: Optional list of earlier group names to scope to.
    """

    kind: Literal["by_cylindrical"] = "by_cylindrical"
    origin: Annotated[
        tuple[float, float, float],
        Field(description="Point on the cylinder axis (3-vector)."),
    ]
    axis: Annotated[
        Literal["x", "y", "z"],
        Field("z", description="Cylinder axis."),
    ]
    r_intervals: Annotated[
        list[float] | None,
        Field(None, description="Radial bin edges (>=0, strictly ascending)."),
    ]
    theta_intervals_deg: Annotated[
        list[float] | None,
        Field(
            None,
            description="Angular bin edges in degrees in [0,360], strictly ascending.",
        ),
    ]
    axial_intervals: Annotated[
        list[float] | None,
        Field(None, description="Bin edges along axis (strictly ascending)."),
    ]
    name_template: Annotated[
        str,
        Field(
            "r{idx}",
            description=(
                "Format string. Placeholders: "
                "{idx} (linear), {ir}, {it}, {iz} (per-axis)."
            ),
        ),
    ]
    restrict_to: Annotated[
        list[str] | None,
        Field(None, description="Optional list of earlier group names to restrict to."),
    ]

    @field_validator("r_intervals")
    def _r_valid(cls, v: list[float] | None) -> list[float] | None:
        if v is None:
            return None
        if len(v) < 2:
            raise ValueError("r_intervals must have at least 2 values")
        if v[0] < 0:
            raise ValueError("r_intervals must be non-negative")
        for i in range(len(v) - 1):
            if v[i] >= v[i + 1]:
                raise ValueError("r_intervals must be strictly ascending")
        return v

    @field_validator("theta_intervals_deg")
    def _theta_valid(cls, v: list[float] | None) -> list[float] | None:
        if v is None:
            return None
        if len(v) < 2:
            raise ValueError("theta_intervals_deg must have at least 2 values")
        if v[0] < 0 or v[-1] > 360:
            raise ValueError("theta_intervals_deg must lie in [0, 360]")
        for i in range(len(v) - 1):
            if v[i] >= v[i + 1]:
                raise ValueError("theta_intervals_deg must be strictly ascending")
        return v

    @field_validator("axial_intervals")
    def _axial_valid(cls, v: list[float] | None) -> list[float] | None:
        if v is None:
            return None
        if len(v) < 2:
            raise ValueError("axial_intervals must have at least 2 values")
        for i in range(len(v) - 1):
            if v[i] >= v[i + 1]:
                raise ValueError("axial_intervals must be strictly ascending")
        return v


def _cells(edges: list[float] | None) -> list[tuple[int, float, float]]:
    """Yield (idx, lo, hi) per cell; None edges -> single (-inf, +inf) cell."""
    if edges is None:
        return [(0, float("-inf"), float("inf"))]
    return [(i, edges[i], edges[i + 1]) for i in range(len(edges) - 1)]


def apply_by_cylindrical(
    spec: ByCylindricalGrouping,
    mesh: LnasFormat,
    allowed: np.ndarray | None,
) -> dict[str, np.ndarray]:
    """Compute (r, theta_deg, axial) per candidate centroid and bin into cells."""
    triangles = mesh.geometry.triangle_vertices
    centroids = np.mean(triangles, axis=1).astype(np.float64)
    n_parent = centroids.shape[0]

    if allowed is not None:
        cand = np.asarray(allowed, dtype=np.int64)
    else:
        cand = np.arange(n_parent, dtype=np.int64)

    if cand.size == 0:
        return {}

    origin = np.asarray(spec.origin, dtype=np.float64)
    rel = centroids[cand] - origin

    a_idx = _AXIS_INDEX[spec.axis]
    p_idx, q_idx = _INPLANE[spec.axis]

    p = rel[:, p_idx]
    q = rel[:, q_idx]
    r = np.sqrt(p * p + q * q)
    theta = np.degrees(np.arctan2(q, p))
    theta = np.where(theta < 0.0, theta + 360.0, theta)
    axial = rel[:, a_idx]

    out: dict[str, np.ndarray] = {}
    linear = 0
    for (ir, rlo, rhi), (it, tlo, thi), (iz, zlo, zhi) in itertools.product(
        _cells(spec.r_intervals),
        _cells(spec.theta_intervals_deg),
        _cells(spec.axial_intervals),
    ):
        # A theta upper edge exactly at 360 must include the (rare) centroid
        # whose normalised theta is exactly 360 due to floating-point quirks.
        thi_eff = float(np.nextafter(thi, np.inf)) if thi == 360.0 else thi
        in_cell = (
            (r >= rlo) & (r < rhi)
            & (theta >= tlo) & (theta < thi_eff)
            & (axial >= zlo) & (axial < zhi)
        )
        cell_idxs = cand[in_cell]
        if cell_idxs.size > 0:
            name = spec.name_template.format(idx=linear, ir=ir, it=it, iz=iz)
            if name in out:
                raise ValueError(
                    f"ByCylindricalGrouping: name_template {spec.name_template!r} "
                    f"produced duplicate group name {name!r}; include "
                    f"{{ir}}/{{it}}/{{iz}} for uniqueness"
                )
            out[name] = cell_idxs
        linear += 1
    return out
