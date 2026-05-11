"""Cm recipe -- per-body moment coefficients.

Same shape as :mod:`cfdmod.core.recipes.cf`, but the input fields are
moment contributions per element (``cm_x``, ``cm_y``, ``cm_z``)
already pre-multiplied by the lever arm. The aggregation is a *sum*
(net moment over the body), not an area-weighted mean.

Lever-arm computation is left to the caller -- the data source must
already carry ``cm_<dir>`` fields. For the end-to-end disk-first
pipeline (mesh attachment, force + moment contribution, statistics),
see the YAML template under ``fixtures/tests/pressure/templates/cm.yaml``.
"""

from __future__ import annotations

__all__ = ["CmRecipeConfig", "cm_pipeline"]

from typing import Literal

from pydantic import BaseModel, ConfigDict

from cfdmod.core.data_source import DataSource, GroupsDataSource
from cfdmod.core.ops.data_source_create.field_series_for_groups import (
    FieldSeriesForGroupsParams,
    field_series_for_groups,
)


class CmRecipeConfig(BaseModel):
    model_config = ConfigDict(frozen=True)

    grouping: str
    directions: list[Literal["x", "y", "z"]] = ["x", "y", "z"]
    prefix: str = "cm"


def cm_pipeline(cfg: CmRecipeConfig):
    def run(ds: DataSource) -> GroupsDataSource:
        result: GroupsDataSource | None = None
        for d in cfg.directions:
            field_in = f"{cfg.prefix}_{d}"
            field_out = f"cm_{d}"
            partial = field_series_for_groups(
                ds,
                FieldSeriesForGroupsParams(
                    grouping=cfg.grouping,
                    field=field_in,
                    agg="sum",
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
