"""Ce recipe -- shape (zoning) coefficients.

Per the odt: a Cp data source plus a planar (zoning) grouping yields a
groups data source whose rows are the zoning regions. Each row is the
area-weighted mean Cp on that region.

The recipe here assumes the zoning grouping is already attached.
The centroid-based assignment lives in the
:func:`cfdmod.core.ops.geometric.zoning_grouping` op; see the YAML
template under ``fixtures/tests/pressure/templates/ce.yaml``.
"""

from __future__ import annotations

__all__ = ["CeRecipeConfig", "ce_pipeline"]

from pydantic import BaseModel, ConfigDict

from cfdmod.core.data_source import DataSource, GroupsDataSource
from cfdmod.core.ops.data_source_create.field_series_for_groups import (
    FieldSeriesForGroupsParams,
    field_series_for_groups,
)


class CeRecipeConfig(BaseModel):
    model_config = ConfigDict(frozen=True)

    grouping: str
    field: str = "cp"
    out: str = "ce"


def ce_pipeline(cfg: CeRecipeConfig):
    def run(ds: DataSource) -> GroupsDataSource:
        return field_series_for_groups(
            ds,
            FieldSeriesForGroupsParams(
                grouping=cfg.grouping,
                field=cfg.field,
                agg="area_weighted_mean",
                out=cfg.out,
            ),
        )

    return run
