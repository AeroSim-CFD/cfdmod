"""Configuration models for the regroup module.

The chain ``RegroupConfig.groupings`` is a superset of
:data:`cfdmod.geometry.grouping.GroupingSpec`. It accepts every standard
grouping kind plus :class:`BySizeRoundedPerComponent`, a regroup-local
fan-out spec that ``run_regroup`` resolves before invoking
:func:`cfdmod.geometry.grouping.apply_groupings`.
"""

from __future__ import annotations

import pathlib
from typing import Annotated, Literal

from pydantic import BaseModel, Field, field_validator

from cfdmod.geometry.grouping.regroup import BySizeRoundedPerComponent, RegroupSpec
from cfdmod.io.geometry.transformation_config import TransformationConfig
from cfdmod.utils import read_yaml

__all__ = [
    "BySizeRoundedPerComponent",
    "RegroupSpec",
    "RegroupConfig",
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
        Literal["per_triangle", "area_weighted_mean", "sliced"],
        Field(
            "area_weighted_mean",
            description=(
                "Per-group HDF5 column policy. "
                "'per_triangle': one column per parent triangle, reordered. "
                "'area_weighted_mean': one aggregated value per group, broadcast. "
                "'sliced': geometrically slice triangles at the cell boundaries "
                "(Ce-style 90-degree cuts) so output triangles never straddle "
                "two cells; per-fragment columns inherit their parent triangle's "
                "value."
            ),
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
