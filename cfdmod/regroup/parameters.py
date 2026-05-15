"""Configuration models for the regroup module.

The chain ``RegroupConfig.groupings`` is a superset of
:data:`cfdmod.geometry.grouping.GroupingSpec`. It accepts every standard
grouping kind plus :class:`BySizeRoundedPerComponent`, a regroup-local
fan-out spec that ``run_regroup`` resolves before invoking
:func:`cfdmod.geometry.grouping.apply_groupings`.
"""

from __future__ import annotations

import pathlib
from typing import Annotated, Literal, Union

from pydantic import BaseModel, Field, field_validator, model_validator

from cfdmod.geometry.grouping.specs import GroupingSpec
from cfdmod.io.geometry.transformation_config import TransformationConfig
from cfdmod.utils import read_yaml

__all__ = [
    "BySizeRoundedPerComponent",
    "RegroupSpec",
    "RegroupConfig",
]


class BySizeRoundedPerComponent(BaseModel):
    """Per-component target-size subdivision with rounded division counts.

    For each group produced by the prior chain, derive per-axis
    ``n_div = max(min_n_div, round(extent / target_size))`` from the
    centroid bounding box restricted to that group, then append a
    :class:`cfdmod.geometry.grouping.ByDivisionsGrouping` with
    ``restrict_to=[group_name]``. Expansion is performed by
    :func:`cfdmod.regroup.run.expand_regroup_chain`; this spec never
    reaches :func:`cfdmod.geometry.grouping.apply_groupings` directly.

    Args:
        kind: Discriminator literal, always ``"by_size_rounded_per_component"``.
        target_size_x, target_size_y, target_size_z: Desired cell size
            along each axis. ``None`` means "do not bin along this axis".
        name_template: Format string for output group names. Available
            placeholders: ``{parent}`` (the source group name) and
            ``{idx}``, ``{ix}``, ``{iy}``, ``{iz}`` (cell indices within
            the parent). The substituted-in template forwarded to
            ``ByDivisionsGrouping`` therefore drops ``{parent}``.
        min_n_div: Floor for the per-axis rounded count (default 1).
        restrict_to: Optional list of earlier group names whose triangles
            define the parent components to fan out over. ``None`` means
            "use the full result of the chain so far".
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
                "Format string for group names. Placeholders: "
                "{parent} (source group), {idx} (linear), {ix}, {iy}, {iz}."
            ),
        ),
    ]
    min_n_div: Annotated[
        int,
        Field(1, ge=1, description="Floor for the per-axis rounded division count."),
    ]
    restrict_to: Annotated[
        list[str] | None,
        Field(None, description="Optional list of earlier group names to fan out over."),
    ]

    @model_validator(mode="after")
    def _at_least_one_target(self) -> "BySizeRoundedPerComponent":
        if (
            self.target_size_x is None
            and self.target_size_y is None
            and self.target_size_z is None
        ):
            raise ValueError(
                "BySizeRoundedPerComponent requires at least one of "
                "target_size_x / target_size_y / target_size_z to be set."
            )
        return self


RegroupSpec = Annotated[
    Union[GroupingSpec, BySizeRoundedPerComponent],
    Field(discriminator="kind"),
]


class RegroupConfig(BaseModel):
    """Top-level config for the regroup pipeline.

    Args:
        groupings: Chain of regroup specs (every standard ``GroupingSpec``
            plus the regroup-local :class:`BySizeRoundedPerComponent`).
            Specs are applied left to right; ``BySizeRoundedPerComponent``
            entries are expanded by ``run_regroup`` against the groups
            produced by the prior prefix of the chain.
        transformation: Optional rigid-body transform applied to a mesh
            *copy* before binning. Output geometry vertices stay in world
            coordinates; only the binning frame moves. Mirrors Ce.
        aggregation: Per-group HDF5 column policy.
            ``"per_triangle"`` reorders the input columns; one output
            column per (parent) triangle.
            ``"area_weighted_mean"`` writes one aggregated value per
            group, broadcast over the post-slice triangles of that group
            (so geometry and timeseries cardinality match for ParaView).
        timeseries_group: HDF5 group name used in the input (under which
            ``t{T}`` datasets live) and reused in the output.
        output_geometry_format: ``"lnas"`` writes only ``geometry.lnas``;
            ``"lnas_and_stl"`` also writes a ``geometry.stl`` companion
            (for quick ParaView lookup; surface labels are lost in STL).
        unassigned_policy: What to do with parent triangles that fall
            in zero groups. ``"drop"`` excludes them; ``"keep_as_unassigned"``
            adds a synthetic ``unassigned`` group/surface.
    """

    groupings: Annotated[
        list[RegroupSpec],
        Field(..., min_length=1, description="Chain of regroup specs."),
    ]
    transformation: Annotated[
        TransformationConfig | None,
        Field(None, description="Optional pre-binning transformation."),
    ]
    aggregation: Annotated[
        Literal["per_triangle", "area_weighted_mean"],
        Field(
            "area_weighted_mean",
            description="Per-group HDF5 column policy.",
        ),
    ]
    timeseries_group: Annotated[
        str,
        Field("cp", description="HDF5 group name in input/output timeseries."),
    ]
    output_geometry_format: Annotated[
        Literal["lnas", "lnas_and_stl"],
        Field("lnas", description="Output mesh format(s)."),
    ]
    unassigned_policy: Annotated[
        Literal["drop", "keep_as_unassigned"],
        Field("drop", description="Policy for parent triangles in zero groups."),
    ]

    @field_validator("groupings")
    def _fan_out_must_be_last(cls, v: list[RegroupSpec]) -> list[RegroupSpec]:
        for i, spec in enumerate(v[:-1]):
            if isinstance(spec, BySizeRoundedPerComponent):
                raise ValueError(
                    "BySizeRoundedPerComponent must be the last spec in the chain "
                    f"(found at position {i} of {len(v)}); downstream specs cannot "
                    "reference its expanded group names yet."
                )
        return v

    @classmethod
    def from_file(cls, path: pathlib.Path) -> "RegroupConfig":
        return cls(**read_yaml(pathlib.Path(path)))
