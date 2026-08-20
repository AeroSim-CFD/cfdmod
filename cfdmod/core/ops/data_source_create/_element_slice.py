"""Shared element-axis slicing for the filter ops.

Slicing a data source along its element axis means slicing four things
consistently: the topology connectivity, the element metadata, every
grouping index, and every field array. Both ``filter_by_grouping`` and
``filter_by_submesh`` need exactly that, differing only in how they pick
the indices.
"""

from __future__ import annotations

__all__ = ["slice_data_source"]

import numpy as np

from cfdmod.adapters.memory import MemoryFieldStore
from cfdmod.core.data_source import DataSource
from cfdmod.core.grouping import Grouping
from cfdmod.core.topology import ElementMeta, Topology


def _slice_topology(topo: Topology | None, idx: np.ndarray) -> Topology | None:
    if topo is None:
        return None
    if topo.cell_type == "point":
        return Topology.points(topo.vertices[idx])
    # triangle / cell: slice connectivity rows; vertices are kept (no
    # remapping). Downstream tools accept dangling vertices; remapping
    # is a separate op (face_cut).
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


def slice_data_source(ds: DataSource, idx: np.ndarray) -> DataSource:
    """Return ``ds`` restricted to the elements ``idx``, in that order.

    ``idx`` is an integer index array into the element axis; it does not
    need to be sorted, so a caller can also use it to reorder the elements
    (that is what ``filter_by_submesh`` does -- the output follows the
    sub-mesh triangle order, not the reference order).
    """
    new_groupings = {
        gname: Grouping(name=gname, indices=g.indices[idx], id_to_label=g.id_to_label)
        for gname, g in ds.groupings.items()
    }
    new_arrays = {fname: ds.fields.read(fname)[idx] for fname in ds.fields.keys()}

    return ds.model_copy(
        update={
            "topology": _slice_topology(ds.topology, idx),
            "elements": _slice_elements(ds.elements, idx),
            "groupings": new_groupings,
            "fields": MemoryFieldStore(new_arrays),
        }
    )
