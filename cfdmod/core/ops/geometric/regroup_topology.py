"""Build a new groups data source by reorganising parent triangles.

The op composes the canonical triangle-grouping pipeline
(:func:`cfdmod.geometry.grouping.apply_groupings`, optionally
pre-expanded via
:func:`cfdmod.geometry.grouping.expand_size_rounded_chain`) with a
field-side aggregation over each produced group. The result is a
:class:`GroupsDataSource` with one row per group; downstream ops can
chain off it like any other data source.

The op intentionally requires non-overlapping groups (each parent
triangle in at most one output group): the result has a single
:class:`Grouping` mapping parent triangles to one group id, and that
mapping must be well defined. For multi-membership use cases (a triangle
in several groups), apply the chain manually and call
``field_series_for_groups`` once per grouping.
"""

from __future__ import annotations

__all__ = ["RegroupTopologyParams", "regroup_topology"]

import pathlib
from typing import ClassVar, Literal

import numpy as np
from lnas import LnasFormat
from pydantic import ConfigDict, Field

from cfdmod.adapters.memory import MemoryFieldStore
from cfdmod.core.data_source import DataSource, GroupsDataSource
from cfdmod.core.field_meta import FieldMeta
from cfdmod.core.grouping import AggregationKind, Grouping, aggregate_rows
from cfdmod.core.ops import OpParams
from cfdmod.core.topology import ElementMeta, Topology
from cfdmod.geometry.grouping import (
    RegroupSpec,
    apply_groupings,
    expand_size_rounded_chain,
)


class RegroupTopologyParams(OpParams):
    """Parameters for :func:`regroup_topology`.

    Attributes:
        mesh: Path to the parent ``.lnas`` mesh. The triangle count must
            match ``ds.n_elements``.
        groupings: Mixed list of canonical
            :class:`~cfdmod.geometry.grouping.GroupingSpec` and
            regroup-only extensions
            (:class:`~cfdmod.geometry.grouping.BySizeRoundedPerComponent`).
            Specs are expanded then applied left-to-right; each parent
            triangle must end up in at most one final group.
        aggregation: Reduction applied per group across each field's
            element axis. ``area_weighted_mean`` requires the parent
            mesh's triangle areas (always available from the lnas).
        grouping_name: Name under which the parent->group mapping is
            stored on the returned data source. Defaults to
            ``"regroup"``.
    """

    model_config = ConfigDict(frozen=True, arbitrary_types_allowed=True)

    kind: Literal["regroup_topology"] = "regroup_topology"
    mesh: str
    groupings: list[RegroupSpec] = Field(default_factory=list)
    aggregation: AggregationKind = "area_weighted_mean"
    grouping_name: str = "regroup"

    chunkable_along: ClassVar[frozenset[str]] = frozenset({"time"})


def _build_parent_grouping(
    n_parent: int,
    output_names: list[str],
    members_per_group: list[np.ndarray],
) -> Grouping:
    """Collapse the per-group triangle lists into a single per-element index array.

    Raises if any parent triangle is claimed by more than one group.
    """
    indices = np.full(n_parent, -1, dtype=np.int32)
    for gid, members in enumerate(members_per_group):
        existing = indices[members]
        clashes = members[existing != -1]
        if clashes.size:
            other = sorted({output_names[int(indices[c])] for c in clashes[:5]})
            raise ValueError(
                f"regroup_topology: parent triangle(s) {clashes[:5].tolist()} "
                f"belong to multiple output groups (e.g. {other} and "
                f"{output_names[gid]!r}); aggregation requires non-overlapping groups"
            )
        indices[members] = gid
    return Grouping(
        name="regroup",
        indices=indices,
        id_to_label={i: name for i, name in enumerate(output_names)},
    )


def regroup_topology(ds: DataSource, p: RegroupTopologyParams) -> GroupsDataSource:
    if ds.topology is None or ds.topology.cell_type != "triangle":
        raise ValueError("regroup_topology requires a triangle (surface) parent")

    lnas = LnasFormat.from_file(pathlib.Path(p.mesh))
    if lnas.geometry.triangles.shape[0] != ds.n_elements:
        raise ValueError(
            f"mesh has {lnas.geometry.triangles.shape[0]} triangles but data source has "
            f"{ds.n_elements} elements."
        )

    if not p.groupings:
        raise ValueError("regroup_topology: groupings list is empty")

    expanded = expand_size_rounded_chain(lnas, list(p.groupings))
    result = apply_groupings(lnas, expanded)

    # Drop scaffolding groups: any group referenced by a later spec's
    # restrict_to is intermediate, not part of the output cardinality.
    referenced: set[str] = set()
    for spec in expanded:
        rt = getattr(spec, "restrict_to", None)
        if rt:
            referenced.update(rt)
    leaf_names = [n for n in result.groups if n not in referenced]
    if not leaf_names:
        raise ValueError("regroup_topology: produced zero non-empty groups")

    output_names = leaf_names
    members_per_group = [np.asarray(result.groups[n], dtype=np.int64) for n in output_names]

    parent_grouping = _build_parent_grouping(ds.n_elements, output_names, members_per_group)

    weights = lnas.geometry.areas if p.aggregation == "area_weighted_mean" else None

    new_arrays: dict[str, np.ndarray] = {}
    new_meta: dict[str, FieldMeta] = {}
    n_groups = len(output_names)
    for fname in ds.fields.keys():
        arr = np.asarray(ds.fields.read(fname), dtype=np.float64)
        is_time = arr.ndim == 2
        n_t = arr.shape[1] if is_time else 0
        out_arr = np.zeros((n_groups, n_t)) if is_time else np.zeros(n_groups)
        for row, members in enumerate(members_per_group):
            out_arr[row] = aggregate_rows(arr, members, p.aggregation, weights)
        new_arrays[fname] = out_arr
        src_meta = ds.field_meta.get(fname)
        new_meta[fname] = (
            FieldMeta(name=fname, unit=src_meta.unit, scale=src_meta.scale)
            if src_meta is not None
            else FieldMeta(name=fname)
        )

    parent_topology = ds.topology
    if parent_topology.n_elements != ds.n_elements:
        # Fall back to constructing parent topology from the lnas directly.
        parent_topology = Topology.triangles(
            lnas.geometry.triangles, lnas.geometry.vertices.astype(np.float64)
        )

    group_grouping = Grouping(
        name=p.grouping_name,
        indices=np.arange(n_groups, dtype=np.int32),
        id_to_label={i: name for i, name in enumerate(output_names)},
    )

    return GroupsDataSource(
        time=ds.time,
        topology=None,
        elements=ElementMeta(),
        parent_topology=parent_topology,
        parent_grouping=parent_grouping,
        groupings={p.grouping_name: group_grouping},
        fields=MemoryFieldStore(new_arrays),
        field_meta=new_meta,
    )
