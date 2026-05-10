"""Data source creation ops -- produce a new index or a new data source.

Per issue #131:

- Statistics (mean, rms, peak, ...) -> a new time-aggregated source.
- Field series for groups -> spatial aggregate per group, attached to a
  groups data source.
- Interpolate (probe extraction, profile interpolation, remeshing).
- Face cut, filter-by-grouping.
- Modal projection / recomposition.
- Integral transforms.

Each op is a pure function ``op(ds_or_container, params) -> new ds``.
The shape difference vs field ops: the result has a *new* element axis,
new time axis, or both. Field ops never reshape.
"""

from __future__ import annotations

__all__ = [
    "StatisticsParams",
    "compute_statistics",
    "FieldSeriesForGroupsParams",
    "field_series_for_groups",
    "FilterByGroupingParams",
    "filter_by_grouping",
]

from cfdmod.core.ops.data_source_create.field_series_for_groups import (
    FieldSeriesForGroupsParams,
    field_series_for_groups,
)
from cfdmod.core.ops.data_source_create.filter_by_grouping import (
    FilterByGroupingParams,
    filter_by_grouping,
)
from cfdmod.core.ops.data_source_create.statistics import StatisticsParams, compute_statistics
