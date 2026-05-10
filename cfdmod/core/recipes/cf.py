"""Cf recipe -- aggregate Cp into per-body force coefficients.

Per the odt::

    container of Cps + grouping -> grouped Cps
    container of grouped Cps    -> aggregate series on groups

Concretely, given a Cp data source with a ``"body"`` grouping
(triangle -> body id), the recipe builds a :class:`GroupsDataSource`
whose fields are the area-weighted-mean Cp on each direction component.

When the input has multiple direction-component fields
(``cp_x``, ``cp_y``, ``cp_z``) the recipe produces one output field per
direction, all on the same groups data source.

The recipe is the small-data analogue of
``cfdmod.pressure.functions.process_Cf``. The legacy disk-first
``run_cf`` keeps producing identical XDMF output until v3.
"""

from __future__ import annotations

__all__ = ["CfRecipeConfig", "cf_pipeline"]

from typing import Literal

from pydantic import BaseModel, ConfigDict

from cfdmod.core.data_source import DataSource, GroupsDataSource
from cfdmod.core.ops.data_source_create.field_series_for_groups import (
    FieldSeriesForGroupsParams,
    field_series_for_groups,
)


class CfRecipeConfig(BaseModel):
    """Cf recipe parameters.

    Attributes:
        grouping: Name of the grouping in ``ds.groupings`` that maps
            triangles to body ids.
        directions: Field-name suffixes per force direction. The recipe
            aggregates ``cp_<dir>`` for each entry. Defaults to the
            three Cartesian components.
        prefix: Source field prefix (``"cp"`` for the standard
            convention).
    """

    model_config = ConfigDict(frozen=True)

    grouping: str
    directions: list[Literal["x", "y", "z"]] = ["x", "y", "z"]
    prefix: str = "cp"


def cf_pipeline(cfg: CfRecipeConfig):
    """Return a single-arg callable producing a :class:`GroupsDataSource`."""

    def run(ds: DataSource) -> GroupsDataSource:
        result: GroupsDataSource | None = None
        for d in cfg.directions:
            field_in = f"{cfg.prefix}_{d}"
            field_out = f"cf_{d}"
            partial = field_series_for_groups(
                ds,
                FieldSeriesForGroupsParams(
                    grouping=cfg.grouping,
                    field=field_in,
                    agg="area_weighted_mean",
                    out=field_out,
                ),
            )
            if result is None:
                result = partial
            else:
                arr = partial.fields.read(field_out)
                meta = partial.field_meta[field_out]
                result = result.with_field(field_out, arr, meta=meta)
        assert result is not None
        return result

    return run
