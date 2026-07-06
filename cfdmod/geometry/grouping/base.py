"""Core types for the triangle-grouping pipeline.

Holds :class:`GroupingResult` and the :func:`apply_groupings` driver. The
per-kind ``apply_*`` functions live under :mod:`cfdmod.geometry.grouping.kinds`
and are dispatched here by spec type.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING

import numpy as np
from lnas import LnasFormat

if TYPE_CHECKING:
    import pandas as pd

from cfdmod.geometry.grouping.kinds.by_connectivity import (
    ByConnectivityGrouping,
    apply_by_connectivity,
)
from cfdmod.geometry.grouping.kinds.by_custom import CustomGrouping, apply_by_custom
from cfdmod.geometry.grouping.kinds.by_cylindrical import (
    ByCylindricalGrouping,
    apply_by_cylindrical,
)
from cfdmod.geometry.grouping.kinds.by_divisions import (
    ByDivisionsGrouping,
    apply_by_divisions,
)
from cfdmod.geometry.grouping.kinds.by_normal import ByNormalGrouping, apply_by_normal
from cfdmod.geometry.grouping.kinds.by_percentile import (
    ByPercentileGrouping,
    apply_by_percentile,
)
from cfdmod.geometry.grouping.kinds.by_plane import ByPlaneGrouping, apply_by_plane
from cfdmod.geometry.grouping.kinds.by_size import BySizeGrouping, apply_by_size
from cfdmod.geometry.grouping.kinds.by_surface import BySurfaceGrouping, apply_by_surface
from cfdmod.geometry.grouping.kinds.by_zoning import ByZoningGrouping, apply_by_zoning
from cfdmod.geometry.grouping.specs import GroupingSpec
from cfdmod.logger import logger


@dataclass(frozen=True)
class GroupingResult:
    """Result of applying a chain of groupings to a parent mesh.

    Triangle indices are into the parent ``LnasFormat.geometry.triangles``
    array (0..parent_n_triangles-1). A triangle may appear in zero, one,
    or many groups.

    Attributes:
        parent_n_triangles: Number of triangles in the parent mesh.
        groups: Mapping of ``group_name -> sorted np.int64 triangle indices``.
    """

    parent_n_triangles: int
    groups: dict[str, np.ndarray] = field(default_factory=dict)

    def membership_long(self) -> pd.DataFrame:
        """Long-form ``(triangle_idx, group_name)`` table.

        One row per ``(triangle, group)`` pair; the table is the natural
        place to express overlapping or absent group membership.

        Returns:
            DataFrame with columns ``triangle_idx`` (int64) and
            ``group_name`` (str). Empty if no groups were produced.
        """
        import pandas as pd

        if not self.groups:
            return pd.DataFrame(
                {
                    "triangle_idx": np.empty(0, dtype=np.int64),
                    "group_name": pd.Series([], dtype="string"),
                }
            )
        tri_arrs = []
        name_arrs = []
        for name, idxs in self.groups.items():
            tri_arrs.append(np.asarray(idxs, dtype=np.int64))
            name_arrs.append(np.full(len(idxs), name, dtype=object))
        return pd.DataFrame(
            {
                "triangle_idx": (
                    np.concatenate(tri_arrs) if tri_arrs else np.empty(0, dtype=np.int64)
                ),
                "group_name": pd.array(np.concatenate(name_arrs), dtype="string"),
            }
        )

    def to_region_idx(self, sep: str = "|", unassigned: str = "") -> np.ndarray:
        """Single-label-per-triangle view, sep-joined when overlapping.

        Triangles in no group are labelled ``unassigned`` (default empty
        string). Useful for legacy code paths that expect one region label
        per triangle.

        Args:
            sep: Separator used when a triangle belongs to several groups.
            unassigned: Label for triangles in no group.

        Returns:
            Object-dtype array of length ``parent_n_triangles``.
        """
        out = np.full(self.parent_n_triangles, unassigned, dtype=object)
        if not self.groups:
            return out
        # Per-triangle list of group names, in insertion order of `groups`.
        per_tri: list[list[str]] = [[] for _ in range(self.parent_n_triangles)]
        for name, idxs in self.groups.items():
            for t in idxs:
                per_tri[int(t)].append(name)
        for t, names in enumerate(per_tri):
            if names:
                out[t] = sep.join(names)
        return out


def apply_groupings(
    mesh: LnasFormat,
    groupings: list[GroupingSpec],
) -> GroupingResult:
    """Apply a chain of grouping specs to a parent mesh.

    Specs are applied left to right. Each spec produces a fresh
    ``name -> indices`` map which is merged into the running result;
    duplicate group names raise ``ValueError`` (no silent merging).

    A spec may carry a ``restrict_to: list[str] | None`` field. When set,
    that spec only considers triangles whose index is in the union of
    the named earlier groups; this is how the legacy
    ``surface -> sub_body`` nesting is expressed.

    Args:
        mesh: Parent mesh.
        groupings: Specs in application order.

    Returns:
        :class:`GroupingResult` over ``mesh``.

    Raises:
        ValueError: If ``groupings`` is empty, two specs produce the same
            group name, or a ``restrict_to`` references an unknown group.
    """
    if not groupings:
        raise ValueError("apply_groupings: groupings list is empty (no-op)")

    parent_n = int(mesh.geometry.triangles.shape[0])
    groups: dict[str, np.ndarray] = {}

    logger.info(
        f"apply_groupings: {len(groupings)} grouping(s) on mesh with "
        f"{parent_n} triangle(s) and {len(mesh.surfaces)} surface(s)"
    )

    for i, spec in enumerate(groupings):
        restrict_to = getattr(spec, "restrict_to", None)
        if restrict_to:
            missing = [n for n in restrict_to if n not in groups]
            if missing:
                raise ValueError(
                    f"apply_groupings[{i}] {type(spec).__name__}: "
                    f"restrict_to references unknown groups: {missing}. "
                    f"Available: {sorted(groups)}"
                )
            mask = np.zeros(parent_n, dtype=bool)
            for n in restrict_to:
                mask[groups[n]] = True
            allowed = np.flatnonzero(mask)
        else:
            allowed = None  # no restriction -> consider all parent triangles

        new_groups = _dispatch(spec, mesh, allowed)

        collisions = [n for n in new_groups if n in groups]
        if collisions:
            raise ValueError(
                f"apply_groupings[{i}] {type(spec).__name__}: "
                f"group name collision with earlier specs: {collisions}"
            )

        logger.info(
            f"  [{i}] {type(spec).__name__} produced {len(new_groups)} group(s): "
            f"{list(new_groups.keys())[:5]}{'...' if len(new_groups) > 5 else ''}"
        )
        groups.update(new_groups)

    return GroupingResult(parent_n_triangles=parent_n, groups=groups)


def _dispatch(
    spec: GroupingSpec,
    mesh: LnasFormat,
    allowed: np.ndarray | None,
) -> dict[str, np.ndarray]:
    if isinstance(spec, BySurfaceGrouping):
        return apply_by_surface(spec, mesh, allowed)
    if isinstance(spec, ByZoningGrouping):
        return apply_by_zoning(spec, mesh, allowed)
    if isinstance(spec, ByDivisionsGrouping):
        return apply_by_divisions(spec, mesh, allowed)
    if isinstance(spec, BySizeGrouping):
        return apply_by_size(spec, mesh, allowed)
    if isinstance(spec, ByConnectivityGrouping):
        return apply_by_connectivity(spec, mesh, allowed)
    if isinstance(spec, ByNormalGrouping):
        return apply_by_normal(spec, mesh, allowed)
    if isinstance(spec, ByPlaneGrouping):
        return apply_by_plane(spec, mesh, allowed)
    if isinstance(spec, ByPercentileGrouping):
        return apply_by_percentile(spec, mesh, allowed)
    if isinstance(spec, ByCylindricalGrouping):
        return apply_by_cylindrical(spec, mesh, allowed)
    if isinstance(spec, CustomGrouping):
        return apply_by_custom(spec, mesh, allowed)
    raise TypeError(f"unknown grouping kind: {type(spec).__name__}")
