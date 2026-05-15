"""Group triangles by the cardinal direction their outward normal aligns with.

For each candidate triangle, the outward normal (right-hand rule on
vertex order) is compared against each requested cardinal direction.
The triangle is assigned to the direction with the largest dot product,
provided the angle between the two does not exceed ``tolerance_deg``.
Triangles whose best-fit angle exceeds the tolerance are not assigned
(consistent with the chain-wide "omit" behavior).

Common use case: split a building's faces into windward / leeward /
roof / sidewall buckets in a single spec.
"""

from __future__ import annotations

from typing import Annotated, Literal

import numpy as np
from lnas import LnasFormat
from pydantic import BaseModel, Field, field_validator

_AXIS_DIRS: dict[str, np.ndarray] = {
    "+x": np.array([1.0, 0.0, 0.0]),
    "-x": np.array([-1.0, 0.0, 0.0]),
    "+y": np.array([0.0, 1.0, 0.0]),
    "-y": np.array([0.0, -1.0, 0.0]),
    "+z": np.array([0.0, 0.0, 1.0]),
    "-z": np.array([0.0, 0.0, -1.0]),
}


class ByNormalGrouping(BaseModel):
    """Group triangles by best-fit cardinal direction of their outward normal.

    Args:
        kind: Discriminator literal, always ``"by_normal"``.
        axes: Cardinal directions to produce buckets for. Subset of
            ``{"+x", "-x", "+y", "-y", "+z", "-z"}``.
        tolerance_deg: Maximum angle (degrees) between a triangle normal
            and its best-fit cardinal direction. With the default
            ``45.0`` every non-degenerate normal lands in exactly one
            bucket; a tighter value (e.g. ``30.0``) excludes oblique
            faces from all buckets.
        name_template: Format string. Placeholder ``{axis}`` is the
            literal axis token (``"+x"`` ... ``"-z"``).
        restrict_to: Optional list of earlier group names to scope to.
    """

    kind: Literal["by_normal"] = "by_normal"
    axes: Annotated[
        list[Literal["+x", "-x", "+y", "-y", "+z", "-z"]],
        Field(
            default_factory=lambda: ["+x", "-x", "+y", "-y", "+z", "-z"],
            description="Cardinal directions to produce buckets for.",
        ),
    ]
    tolerance_deg: Annotated[
        float,
        Field(
            45.0,
            gt=0.0,
            le=90.0,
            description="Max angle (deg) between a normal and its best-fit direction.",
        ),
    ]
    name_template: Annotated[
        str,
        Field("n_{axis}", description="Format string; placeholder: {axis}."),
    ]
    restrict_to: Annotated[
        list[str] | None,
        Field(None, description="Optional list of earlier group names to restrict to."),
    ]

    @field_validator("axes")
    def _no_dup_axes(cls, v: list[str]) -> list[str]:
        if not v:
            raise ValueError("axes must not be empty")
        if len(v) != len(set(v)):
            raise ValueError("axes must not contain duplicates")
        return v


def apply_by_normal(
    spec: ByNormalGrouping,
    mesh: LnasFormat,
    allowed: np.ndarray | None,
) -> dict[str, np.ndarray]:
    """Compute per-triangle normals and bucket by best-fit cardinal direction."""
    triangles = mesh.geometry.triangle_vertices  # (n_tri, 3, 3)
    n_parent = triangles.shape[0]

    if allowed is not None:
        cand = np.asarray(allowed, dtype=np.int64)
    else:
        cand = np.arange(n_parent, dtype=np.int64)

    if cand.size == 0:
        return {}

    cand_tris = triangles[cand].astype(np.float64)
    e1 = cand_tris[:, 1] - cand_tris[:, 0]
    e2 = cand_tris[:, 2] - cand_tris[:, 0]
    normals = np.cross(e1, e2)
    norms = np.linalg.norm(normals, axis=1, keepdims=True)
    safe = np.where(norms > 0, norms, 1.0)
    unit_normals = normals / safe

    dirs = np.stack([_AXIS_DIRS[a] for a in spec.axes], axis=0)  # (k, 3)
    cosines = unit_normals @ dirs.T  # (n_cand, k)
    cos_thresh = float(np.cos(np.deg2rad(spec.tolerance_deg)))

    best = np.argmax(cosines, axis=1)
    best_cos = cosines[np.arange(cosines.shape[0]), best]
    valid = (best_cos >= cos_thresh) & (norms.flatten() > 0.0)

    out: dict[str, np.ndarray] = {}
    for k, axis_name in enumerate(spec.axes):
        mask = valid & (best == k)
        if not mask.any():
            continue
        name = spec.name_template.format(axis=axis_name)
        if name in out:
            raise ValueError(
                f"ByNormalGrouping: name_template {spec.name_template!r} produced "
                f"duplicate group name {name!r}; include {{axis}} for uniqueness"
            )
        out[name] = np.sort(cand[mask])
    return out
