"""Group triangles by axis-aligned spatial bins of their centroids.

This is the generalisation of the legacy ``ZoningModel`` in
``cfdmod.pressure.parameters``: a regular Cartesian grid in
``(x_intervals, y_intervals, z_intervals)``, each cell becoming one
named group. Triangles whose centroid falls in no cell are not added to
any group.

When composed with an earlier ``BySurfaceGrouping`` via ``restrict_to``,
this reproduces today's ``surface -> sub_body`` nesting.
"""

from __future__ import annotations

import itertools
from typing import Annotated, Literal

import numpy as np
from lnas import LnasFormat
from pydantic import BaseModel, Field, field_validator


class ByZoningGrouping(BaseModel):
    """Axis-aligned centroid binning into a Cartesian grid of regions.

    Args:
        kind: Discriminator literal, always ``"by_zoning"``.
        x_intervals, y_intervals, z_intervals: Strictly ascending,
            non-repeating bin edges. ``[-inf, inf]`` (the default) means
            "do not bin along this axis"; the axis contributes a single
            cell.
        name_template: Format string for group names. Available
            placeholders: ``{idx}`` (linear region index, 0-based),
            ``{ix}``, ``{iy}``, ``{iz}`` (per-axis cell indices).
        restrict_to: Optional list of earlier group names; when set,
            only triangles in (the union of) those groups are considered.
            Triangles outside the restriction are not assigned by this
            spec.
    """

    kind: Literal["by_zoning"] = "by_zoning"
    x_intervals: Annotated[
        list[float],
        Field(
            default_factory=lambda: [float("-inf"), float("inf")],
            description="X axis bin edges (strictly ascending, non-repeating)",
        ),
    ]
    y_intervals: Annotated[
        list[float],
        Field(
            default_factory=lambda: [float("-inf"), float("inf")],
            description="Y axis bin edges (strictly ascending, non-repeating)",
        ),
    ]
    z_intervals: Annotated[
        list[float],
        Field(
            default_factory=lambda: [float("-inf"), float("inf")],
            description="Z axis bin edges (strictly ascending, non-repeating)",
        ),
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

    @field_validator("x_intervals", "y_intervals", "z_intervals")
    def _validate_intervals(cls, v: list[float]) -> list[float]:
        if not v:
            return [float("-inf"), float("inf")]
        if len(v) < 2:
            raise ValueError("interval must have at least 2 values")
        if len(v) != len(set(v)):
            raise ValueError("interval values must not repeat")
        for i in range(len(v) - 1):
            if v[i] >= v[i + 1]:
                raise ValueError("interval values must be strictly ascending")
        return v


def _regions(
    spec: ByZoningGrouping,
) -> list[tuple[int, int, int, int, tuple[float, float, float], tuple[float, float, float]]]:
    """Enumerate (linear_idx, ix, iy, iz, lower, upper) for every cell."""
    x_cells = [
        (spec.x_intervals[i], spec.x_intervals[i + 1]) for i in range(len(spec.x_intervals) - 1)
    ]
    y_cells = [
        (spec.y_intervals[i], spec.y_intervals[i + 1]) for i in range(len(spec.y_intervals) - 1)
    ]
    z_cells = [
        (spec.z_intervals[i], spec.z_intervals[i + 1]) for i in range(len(spec.z_intervals) - 1)
    ]
    out = []
    linear = 0
    for (ix, (xlo, xhi)), (iy, (ylo, yhi)), (iz, (zlo, zhi)) in itertools.product(
        enumerate(x_cells), enumerate(y_cells), enumerate(z_cells)
    ):
        out.append((linear, ix, iy, iz, (xlo, ylo, zlo), (xhi, yhi, zhi)))
        linear += 1
    return out


def apply_by_zoning(
    spec: ByZoningGrouping,
    mesh: LnasFormat,
    allowed: np.ndarray | None,
) -> dict[str, np.ndarray]:
    """Bin triangle centroids into the Cartesian cells declared by ``spec``.

    Args:
        spec: The grouping spec.
        mesh: Parent mesh; uses ``mesh.geometry.triangle_vertices``.
        allowed: Optional sorted parent-triangle indices to restrict the
            binning to. None = consider all triangles.

    Returns:
        ``dict[group_name, sorted int64 parent triangle indices]``. Empty
        cells are omitted from the result.

    Raises:
        ValueError: If two cells produce the same group name (template
            does not disambiguate enough).
    """
    triangles = mesh.geometry.triangle_vertices  # (n_tri, 3, 3)
    centroids = np.mean(triangles, axis=1)  # (n_tri, 3)

    if allowed is not None:
        candidate_idxs = np.asarray(allowed, dtype=np.int64)
    else:
        candidate_idxs = np.arange(centroids.shape[0], dtype=np.int64)

    candidate_centroids = centroids[candidate_idxs]

    out: dict[str, np.ndarray] = {}
    for linear, ix, iy, iz, lo, hi in _regions(spec):
        ll = np.asarray(lo, dtype=np.float64)
        ur = np.asarray(hi, dtype=np.float64)
        in_cell = np.all((candidate_centroids >= ll) & (candidate_centroids < ur), axis=1)
        cell_idxs = candidate_idxs[in_cell]
        if cell_idxs.size == 0:
            continue
        name = spec.name_template.format(idx=linear, ix=ix, iy=iy, iz=iz)
        if name in out:
            raise ValueError(
                f"ByZoningGrouping: name_template {spec.name_template!r} produced "
                f"duplicate group name {name!r}; include {{ix}}/{{iy}}/{{iz}} for uniqueness"
            )
        out[name] = cell_idxs
    return out
