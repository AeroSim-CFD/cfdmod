"""Filter the elements of a data source by group membership.

Given a :class:`DataSource` and the name of an existing grouping, keep
only the elements whose group id is in ``keep`` (or all but those in
``drop``). The output is a new data source with a sliced topology,
sliced element metadata, sliced fields, and sliced groupings (every
existing grouping has its index trimmed accordingly).
"""

from __future__ import annotations

__all__ = ["FilterByGroupingParams", "filter_by_grouping"]

from typing import ClassVar, Literal

import numpy as np

from cfdmod.adapters.memory import MemoryFieldStore
from cfdmod.core.data_source import DataSource
from cfdmod.core.grouping import Grouping
from cfdmod.core.ops import OpParams
from cfdmod.core.topology import ElementMeta, Topology


class FilterByGroupingParams(OpParams):
    """Parameters for :func:`filter_by_grouping`.

    Exactly one of ``keep`` or ``drop`` must be set.

    Attributes:
        grouping: Name of the grouping in ``ds.groupings`` to filter by.
        keep: Group ids to retain.
        drop: Group ids to remove.
    """

    kind: Literal["filter_by_grouping"] = "filter_by_grouping"
    grouping: str
    keep: list[int] | None = None
    drop: list[int] | None = None

    chunkable_along: ClassVar[frozenset[str]] = frozenset({"time"})


def _select_elements(grouping: Grouping, p: FilterByGroupingParams) -> np.ndarray:
    if (p.keep is None) == (p.drop is None):
        raise ValueError("filter_by_grouping requires exactly one of keep or drop")
    if p.keep is not None:
        keep_set = set(int(g) for g in p.keep)
        mask = np.array([int(i) in keep_set for i in grouping.indices], dtype=bool)
    else:
        drop_set = set(int(g) for g in p.drop)
        mask = np.array([int(i) not in drop_set for i in grouping.indices], dtype=bool)
    return np.flatnonzero(mask)


def _slice_topology(topo: Topology | None, idx: np.ndarray) -> Topology | None:
    if topo is None:
        return None
    if topo.cell_type == "point":
        return Topology.points(topo.vertices[idx])
    # triangle / cell: slice connectivity rows; vertices are kept (no
    # remapping). Downstream tools accept dangling vertices; remapping
    # is a separate op (Phase 5: face_cut).
    return Topology(
        cell_type=topo.cell_type,
        connectivity=topo.connectivity[idx],
        vertices=topo.vertices,
    )


def _slice_elements(em: ElementMeta, idx: np.ndarray) -> ElementMeta:
    return ElementMeta(
        position=em.position[idx] if em.position is not None else None,
        area=em.area[idx] if em.area is not None else None,
        volume=em.volume[idx] if em.volume is not None else None,
        normal=em.normal[idx] if em.normal is not None else None,
    )


def filter_by_grouping(ds: DataSource, p: FilterByGroupingParams) -> DataSource:
    if p.grouping not in ds.groupings:
        raise KeyError(f"grouping {p.grouping!r} not found on data source")
    grouping = ds.groupings[p.grouping]

    keep_idx = _select_elements(grouping, p)
    if keep_idx.size == 0:
        raise ValueError(
            f"filter_by_grouping selects 0 elements (grouping={p.grouping!r}, "
            f"keep={p.keep}, drop={p.drop})"
        )

    new_topology = _slice_topology(ds.topology, keep_idx)
    new_elements = _slice_elements(ds.elements, keep_idx)
    new_groupings = {
        gname: Grouping(name=gname, indices=g.indices[keep_idx], id_to_label=g.id_to_label)
        for gname, g in ds.groupings.items()
    }

    new_arrays: dict[str, np.ndarray] = {}
    for fname in ds.fields.keys():
        arr = ds.fields.read(fname)
        new_arrays[fname] = arr[keep_idx]

    return ds.model_copy(
        update={
            "topology": new_topology,
            "elements": new_elements,
            "groupings": new_groupings,
            "fields": MemoryFieldStore(new_arrays),
        }
    )
