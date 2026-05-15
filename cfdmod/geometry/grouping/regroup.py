"""Regroup-pipeline extensions to the canonical triangle-grouping pipeline.

Holds spec kinds that are *not* directly dispatchable by
:func:`apply_groupings` because they require materialising an earlier
prefix of the chain before the spec is fully known. The expansion pass
:func:`expand_size_rounded_chain` rewrites such specs into a flat
``list[GroupingSpec]`` that the canonical pipeline can apply.

Currently this module exposes :class:`BySizeRoundedPerComponent`, used
by the regroup op to subdivide each parent group independently into a
round-to-nearest cell count derived from the parent's own centroid bbox
and per-axis target sizes.
"""

from __future__ import annotations

__all__ = [
    "BySizeRoundedPerComponent",
    "RegroupSpec",
    "expand_size_rounded_chain",
]

from typing import Annotated, Literal, Union

import numpy as np
from lnas import LnasFormat
from pydantic import BaseModel, Field

from cfdmod.geometry.grouping.base import apply_groupings
from cfdmod.geometry.grouping.kinds.by_divisions import ByDivisionsGrouping
from cfdmod.geometry.grouping.specs import GroupingSpec


class BySizeRoundedPerComponent(BaseModel):
    """Per-parent-group round-to-nearest size-based subdivision.

    Expanded by :func:`expand_size_rounded_chain` before
    :func:`apply_groupings` is called: for each group produced by the
    prior chain, derive per-axis ``n_div = max(min_n_div, round(extent
    / target))`` from the restricted centroid bbox, then append a
    :class:`ByDivisionsGrouping` with ``restrict_to=[parent_name]`` to
    the expanded chain.

    Args:
        kind: Discriminator literal, always
            ``"by_size_rounded_per_component"``.
        target_size_x, target_size_y, target_size_z: Approximate cell
            size along each axis. ``None`` (the default) means "do not
            bin along this axis"; the axis contributes a single cell.
        name_template: Format string for emitted group names.
            ``{parent}`` is replaced with the parent group's name and
            ``{idx}``, ``{ix}``, ``{iy}``, ``{iz}`` are forwarded to
            the inner :class:`ByDivisionsGrouping`.
        min_n_div: Floor for the rounded division count per axis;
            defaults to 1.
    """

    kind: Literal["by_size_rounded_per_component"] = "by_size_rounded_per_component"
    target_size_x: Annotated[
        float | None,
        Field(None, gt=0.0, description="Target cell size along x; None = no x binning."),
    ]
    target_size_y: Annotated[
        float | None,
        Field(None, gt=0.0, description="Target cell size along y; None = no y binning."),
    ]
    target_size_z: Annotated[
        float | None,
        Field(None, gt=0.0, description="Target cell size along z; None = no z binning."),
    ]
    name_template: Annotated[
        str,
        Field(
            "{parent}_r{idx}",
            description=(
                "Group-name template. Placeholders: {parent} (source "
                "group name), {idx} (linear cell index), {ix}/{iy}/{iz}."
            ),
        ),
    ]
    min_n_div: Annotated[
        int,
        Field(1, ge=1, description="Floor for the rounded division count."),
    ]


RegroupSpec = Annotated[
    Union[GroupingSpec, BySizeRoundedPerComponent],
    Field(discriminator="kind"),
]
"""Discriminated union of canonical :data:`GroupingSpec` plus the
regroup-only extensions; the input shape consumed by the regroup op
before expansion."""


def _round_half_up(value: float) -> int:
    """``int(value + 0.5)`` -- half-up, deterministic across versions.

    Python's built-in ``round`` uses banker's rounding which surprises
    users who think of "round 0.5 up". Regroup users explicitly want
    "extent 4.5 with target 3 -> 2 cells", not 1.
    """
    return int(np.floor(value + 0.5))


def _n_div_for_axis(extent: float, target: float | None, min_n_div: int) -> int | None:
    if target is None:
        return None
    return max(min_n_div, _round_half_up(extent / target))


def expand_size_rounded_chain(
    mesh: LnasFormat,
    specs: list,
) -> list:
    """Rewrite a regroup chain into a flat list of canonical grouping specs.

    Walks ``specs`` left-to-right. Each :data:`GroupingSpec` is appended
    unchanged. Each :class:`BySizeRoundedPerComponent` is materialised
    against the prior expanded prefix (a fresh
    :func:`apply_groupings` call) so the per-parent-group bounding
    boxes can be computed; one
    :class:`~cfdmod.geometry.grouping.ByDivisionsGrouping` is then
    appended per parent group with
    ``restrict_to=[parent_name]``.

    Args:
        mesh: Parent mesh used for the prefix expansions.
        specs: Mixed list of :data:`GroupingSpec` /
            :class:`BySizeRoundedPerComponent`.

    Returns:
        A flat ``list[GroupingSpec]`` ready to pass to
        :func:`apply_groupings`. Order is preserved; expansions are
        inserted at the position of the original spec.

    Raises:
        ValueError: If a :class:`BySizeRoundedPerComponent` is the very
            first spec (there is no prior chain to derive parent
            groups from).
    """
    expanded: list = []
    for i, spec in enumerate(specs):
        if not isinstance(spec, BySizeRoundedPerComponent):
            expanded.append(spec)
            continue

        if not expanded:
            raise ValueError(
                "expand_size_rounded_chain: BySizeRoundedPerComponent at "
                f"position {i} has no prior chain to derive parent groups from"
            )

        prefix_result = apply_groupings(mesh, expanded)
        parent_groups = prefix_result.groups

        triangles = mesh.geometry.triangle_vertices
        centroids = np.mean(triangles, axis=1)

        for parent_name, parent_idxs in parent_groups.items():
            if parent_idxs.size == 0:
                continue
            cand = centroids[parent_idxs]
            lo = cand.min(axis=0)
            hi = cand.max(axis=0)

            n_div_x = _n_div_for_axis(float(hi[0] - lo[0]), spec.target_size_x, spec.min_n_div)
            n_div_y = _n_div_for_axis(float(hi[1] - lo[1]), spec.target_size_y, spec.min_n_div)
            n_div_z = _n_div_for_axis(float(hi[2] - lo[2]), spec.target_size_z, spec.min_n_div)

            inner_template = spec.name_template.replace("{parent}", parent_name)

            expanded.append(
                ByDivisionsGrouping(
                    n_div_x=n_div_x,
                    n_div_y=n_div_y,
                    n_div_z=n_div_z,
                    name_template=inner_template,
                    restrict_to=[parent_name],
                )
            )

    return expanded
