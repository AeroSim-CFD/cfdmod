"""Group triangle centroids by signed distance from an oriented plane.

Generalises :class:`ByZoningGrouping` to non-axis-aligned half-space
splits. The plane is defined by a ``point`` on it and a ``normal``
direction (auto-normalised); each centroid's signed distance along the
normal is binned via the 1D ``intervals`` partition. The default
``[-inf, 0, inf]`` simply splits the body into the two half-spaces on
either side of the plane.

Useful for buildings rotated in the mesh frame, oblique splits along an
inflow direction, or any partition that ``ByZoning`` cannot express
because its bin walls are not axis-aligned.
"""

from __future__ import annotations

from typing import Annotated, Literal

import numpy as np
from lnas import LnasFormat
from pydantic import BaseModel, Field, field_validator


class ByPlaneGrouping(BaseModel):
    """Bin triangle centroids by signed distance from an oriented plane.

    Args:
        kind: Discriminator literal, always ``"by_plane"``.
        point: A point on the plane (3-vector).
        normal: Plane normal (3-vector, auto-normalised; must be non-zero).
        intervals: Strictly ascending bin edges along ``normal`` (signed
            distances). Default ``[-inf, 0.0, inf]`` (two half-spaces).
        name_template: Format string. Placeholder ``{idx}``.
        restrict_to: Optional list of earlier group names to scope to.
    """

    kind: Literal["by_plane"] = "by_plane"
    point: Annotated[
        tuple[float, float, float],
        Field(description="A point on the plane (3-vector)."),
    ]
    normal: Annotated[
        tuple[float, float, float],
        Field(description="Plane normal (3-vector, auto-normalised; non-zero)."),
    ]
    intervals: Annotated[
        list[float],
        Field(
            default_factory=lambda: [float("-inf"), 0.0, float("inf")],
            description="Signed-distance bin edges along normal; strictly ascending.",
        ),
    ]
    name_template: Annotated[
        str,
        Field("r{idx}", description="Format string; placeholder: {idx}."),
    ]
    restrict_to: Annotated[
        list[str] | None,
        Field(None, description="Optional list of earlier group names to restrict to."),
    ]

    @field_validator("normal")
    def _normal_nonzero(
        cls, v: tuple[float, float, float]
    ) -> tuple[float, float, float]:
        if float(np.linalg.norm(v)) == 0.0:
            raise ValueError("normal must be a non-zero vector")
        return v

    @field_validator("intervals")
    def _intervals_ascending(cls, v: list[float]) -> list[float]:
        if len(v) < 2:
            raise ValueError("intervals must have at least 2 values")
        if len(v) != len(set(v)):
            raise ValueError("interval values must not repeat")
        for i in range(len(v) - 1):
            if v[i] >= v[i + 1]:
                raise ValueError("interval values must be strictly ascending")
        return v


def apply_by_plane(
    spec: ByPlaneGrouping,
    mesh: LnasFormat,
    allowed: np.ndarray | None,
) -> dict[str, np.ndarray]:
    """Project candidate centroids onto the plane normal and bin the offsets."""
    triangles = mesh.geometry.triangle_vertices
    centroids = np.mean(triangles, axis=1)
    n_parent = centroids.shape[0]

    if allowed is not None:
        cand = np.asarray(allowed, dtype=np.int64)
    else:
        cand = np.arange(n_parent, dtype=np.int64)

    if cand.size == 0:
        return {}

    point = np.asarray(spec.point, dtype=np.float64)
    normal = np.asarray(spec.normal, dtype=np.float64)
    normal = normal / np.linalg.norm(normal)

    cand_centroids = centroids[cand].astype(np.float64)
    signed = (cand_centroids - point) @ normal  # (n_cand,)

    out: dict[str, np.ndarray] = {}
    for i in range(len(spec.intervals) - 1):
        lo, hi = spec.intervals[i], spec.intervals[i + 1]
        in_cell = (signed >= lo) & (signed < hi)
        cell_idxs = cand[in_cell]
        if cell_idxs.size == 0:
            continue
        name = spec.name_template.format(idx=i)
        if name in out:
            raise ValueError(
                f"ByPlaneGrouping: name_template {spec.name_template!r} produced "
                f"duplicate group name {name!r}; include {{idx}} for uniqueness"
            )
        out[name] = cell_idxs
    return out
