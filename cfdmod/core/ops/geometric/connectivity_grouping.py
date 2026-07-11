"""Assign triangles to regions by connected component (shared-edge graph).

Each connected component of the mesh's triangle-shared-edge graph becomes
one region. This is the topological alternative to the centroid-based
:func:`zoning_grouping`: it needs no axis projection or axis-aligned
bounding boxes, so it partitions non-rectangular, rotated or irregular
geometries (cylindrical tanks, L-shaped buildings, container packs) into
one region per physically distinct body.

The partitioning itself is delegated to the canonical triangle-grouping
pipeline in :mod:`cfdmod.geometry.grouping` (a
:class:`ByConnectivityGrouping` spec applied via :func:`apply_groupings`);
this op only adapts the result into the per-element index array carried by
a v3 :class:`Grouping`. Region ids are 0-indexed integers ordered by
descending triangle count (region ``0`` is the largest component), matching
the ``cc{idx}`` names emitted by the spec. Components smaller than
``min_triangles`` are dropped; their triangles stay ``-1`` (ungrouped).

The resulting grouping is a drop-in replacement for the one produced by
:func:`body_grouping` when feeding the Cf/Cm/Ce recipes: point the recipe's
``grouping`` config at this op's ``name``.
"""

from __future__ import annotations

__all__ = ["ConnectivityGroupingParams", "connectivity_grouping"]

import pathlib
from typing import ClassVar, Literal

import numpy as np
from lnas import LnasFormat
from pydantic import ConfigDict

from cfdmod.core.data_source import DataSource
from cfdmod.core.grouping import Grouping
from cfdmod.core.ops import OpParams
from cfdmod.geometry.grouping import ByConnectivityGrouping, apply_groupings


class ConnectivityGroupingParams(OpParams):
    """Parameters for :func:`connectivity_grouping`.

    Attributes:
        mesh: Path to a ``.lnas`` file. Connectivity is computed from the
            mesh triangles so this op is independent of upstream
            :func:`mesh_attach`.
        min_triangles: Components with fewer triangles than this are
            dropped (treated as mesh debris); their triangles stay ``-1``.
            Defaults to ``1`` (keep everything).
        name: Grouping name to attach. Defaults to ``"body"`` so the result
            can feed the Cf/Cm/Ce recipes in place of :func:`body_grouping`.
    """

    model_config = ConfigDict(frozen=True)

    kind: Literal["connectivity_grouping"] = "connectivity_grouping"
    mesh: str
    min_triangles: int = 1
    name: str = "body"

    chunkable_along: ClassVar[frozenset[str]] = frozenset({"time"})


def connectivity_grouping(ds: DataSource, p: ConnectivityGroupingParams) -> DataSource:
    lnas = LnasFormat.from_file(pathlib.Path(p.mesh))
    if lnas.geometry.triangles.shape[0] != ds.n_elements:
        raise ValueError(
            f"mesh has {lnas.geometry.triangles.shape[0]} triangles but data source has "
            f"{ds.n_elements} elements."
        )

    spec = ByConnectivityGrouping(min_triangles=p.min_triangles, name_template="cc{idx}")
    result = apply_groupings(lnas, [spec])

    indices = np.full(ds.n_elements, -1, dtype=np.int32)
    id_to_label: dict[int, str] = {}
    for name, tris in result.groups.items():
        region_id = int(name.removeprefix("cc"))
        indices[tris] = region_id
        id_to_label[region_id] = name

    grouping = Grouping(name=p.name, indices=indices, id_to_label=id_to_label)
    return ds.with_grouping(grouping)
