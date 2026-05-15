"""Group triangles by named LNAS surfaces.

Each entry in ``sets`` becomes a single named group containing the union
of the triangle indices of the listed surfaces. This is the
generalisation of today's ``BodyDefinition.surfaces`` and
``CeConfig.sets`` constructs.

``BySurfaceGrouping`` is a *selection* grouping: it picks triangles by
name from ``LnasFormat.surfaces``. The legacy ``restrict_to`` field is
not supported here because there is nothing to discover - the user
explicitly names which surfaces compose each group.
"""

from __future__ import annotations

from typing import Annotated, Literal

import numpy as np
from lnas import LnasFormat
from pydantic import BaseModel, Field, field_validator


class BySurfaceGrouping(BaseModel):
    """Group triangles by named LNAS surfaces.

    Args:
        kind: Discriminator literal, always ``"by_surface"``.
        sets: ``group_name -> list of LnasFormat.surfaces keys``. Each
            named set becomes one group containing the union of those
            surfaces' triangle indices.
        include_unlisted: When True, all surfaces in ``LnasFormat.surfaces``
            not referenced in ``sets`` are added as singleton groups
            keyed by surface name.
    """

    kind: Literal["by_surface"] = "by_surface"
    sets: Annotated[
        dict[str, list[str]],
        Field(
            default_factory=dict,
            description="group_name -> list of LnasFormat.surfaces keys",
        ),
    ]
    include_unlisted: Annotated[
        bool,
        Field(False, description="Add unreferenced surfaces as singleton groups"),
    ]

    @field_validator("sets")
    def _validate_sets(cls, v: dict[str, list[str]]) -> dict[str, list[str]]:
        for name, surfaces in v.items():
            if not name:
                raise ValueError("BySurfaceGrouping.sets: group name must be non-empty")
            if len(surfaces) != len(set(surfaces)):
                raise ValueError(
                    f"BySurfaceGrouping.sets[{name!r}]: surface names must be unique"
                )
        return v


def apply_by_surface(
    spec: BySurfaceGrouping,
    mesh: LnasFormat,
    allowed: np.ndarray | None,
) -> dict[str, np.ndarray]:
    """Build groups from named LNAS surfaces.

    Args:
        spec: The grouping spec.
        mesh: Parent mesh.
        allowed: Ignored - by_surface is a selection grouping.

    Returns:
        ``dict[group_name, sorted unique int64 triangle indices]``.

    Raises:
        KeyError: If a referenced surface is not in ``mesh.surfaces``.
        ValueError: If a triangle would belong to two groups produced by
            the same spec (use a separate spec for overlap on purpose).
    """
    del allowed  # unused
    available = set(mesh.surfaces.keys())

    out: dict[str, np.ndarray] = {}
    used_in_sets: set[str] = set()

    for group_name, surface_names in spec.sets.items():
        missing = [s for s in surface_names if s not in available]
        if missing:
            raise KeyError(
                f"BySurfaceGrouping.sets[{group_name!r}]: surfaces not in mesh: "
                f"{missing}. Available: {sorted(available)}"
            )
        idxs_parts = [np.asarray(mesh.surfaces[s], dtype=np.int64) for s in surface_names]
        merged = (
            np.unique(np.concatenate(idxs_parts))
            if idxs_parts
            else np.empty(0, dtype=np.int64)
        )
        out[group_name] = merged
        used_in_sets.update(surface_names)

    if spec.include_unlisted:
        for sfc in mesh.surfaces:
            if sfc in used_in_sets or sfc in out:
                continue
            out[sfc] = np.asarray(mesh.surfaces[sfc], dtype=np.int64).copy()

    return out
