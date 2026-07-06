"""Spatial aggregation of a field per group -> :class:`GroupsDataSource`.

Given a parent :class:`SurfaceDataSource` (or any data source with a
grouping), reduce a field over each group's elements to produce a new
:class:`GroupsDataSource`. The output's element axis is the *group*
index, not the original element index. Time is preserved.

Aggregations supported: ``mean``, ``sum``, ``area_weighted_mean``
(requires ``elements.area``), ``max``, ``min``.
"""

from __future__ import annotations

__all__ = ["FieldSeriesForGroupsParams", "field_series_for_groups", "AGG_KINDS"]

from typing import ClassVar, Literal

import numpy as np

from cfdmod.adapters.memory import MemoryFieldStore
from cfdmod.core.data_source import DataSource, GroupsDataSource
from cfdmod.core.field_meta import FieldMeta
from cfdmod.core.grouping import AggregationKind, Grouping, aggregate_rows, groups_in
from cfdmod.core.ops import OpParams
from cfdmod.core.topology import ElementMeta

# Backwards-compatible alias; the canonical name lives in cfdmod.core.grouping.
AGG_KINDS = AggregationKind


class FieldSeriesForGroupsParams(OpParams):
    """Parameters for :func:`field_series_for_groups`.

    Attributes:
        grouping: Name of the grouping in ``ds.groupings`` to reduce
            over.
        field: Source field to reduce. Defaults to ``"pressure"``.
        agg: Aggregation kind.
        out: Output field name on the resulting groups source. Defaults
            to ``field``.
    """

    kind: Literal["field_series_for_groups"] = "field_series_for_groups"
    grouping: str
    field: str = "pressure"
    agg: AGG_KINDS = "mean"
    out: str | None = None

    chunkable_along: ClassVar[frozenset[str]] = frozenset({"time"})


def field_series_for_groups(ds: DataSource, p: FieldSeriesForGroupsParams) -> GroupsDataSource:
    if p.grouping not in ds.groupings:
        raise KeyError(f"grouping {p.grouping!r} not found on data source")
    if ds.topology is None or ds.topology.cell_type != "triangle":
        raise ValueError("field_series_for_groups requires a triangle (surface) parent")

    grouping = ds.groupings[p.grouping]
    arr = np.asarray(ds.fields.read(p.field), dtype=np.float64)
    weights = ds.elements.area

    group_ids = groups_in(grouping)
    n_groups = int(group_ids.size)

    is_time = arr.ndim == 2
    n_t = arr.shape[1] if is_time else 0

    out_arr = np.zeros((n_groups, n_t)) if is_time else np.zeros(n_groups)
    for row, gid in enumerate(group_ids):
        members = np.flatnonzero(grouping.indices == gid)
        out_arr[row] = aggregate_rows(arr, members, p.agg, weights)

    target = p.out or p.field
    src_meta = ds.field_meta.get(p.field)
    out_meta = (
        FieldMeta(name=target, unit=src_meta.unit, scale=src_meta.scale)
        if src_meta is not None
        else FieldMeta(name=target)
    )

    # Group-level grouping mapping: identity (group i -> group i) so that
    # downstream consumers can chain operations off the new source.
    group_grouping = Grouping(
        name=p.grouping,
        indices=np.arange(n_groups, dtype=np.int32),
        id_to_label={int(i): grouping.label(int(gid)) for i, gid in enumerate(group_ids)},
    )

    return GroupsDataSource(
        time=ds.time,
        topology=None,
        elements=ElementMeta(),
        parent_topology=ds.topology,
        parent_grouping=grouping,
        groupings={p.grouping: group_grouping},
        fields=MemoryFieldStore({target: out_arr}),
        field_meta={target: out_meta},
    )
