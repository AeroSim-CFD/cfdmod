"""Build a body grouping on a SurfaceDataSource from a mesh + bodies dict.

A "body" is a named subset of the mesh's surfaces. Given:

- a SurfaceDataSource whose triangle count matches the mesh,
- a path to the ``.lnas`` mesh,
- a dict ``{body_name: [surface_names]}`` (empty list = all surfaces),

returns the source with a :class:`Grouping` attached under the
configured name (default ``"body"``). Triangles not in any body get
``-1``. Triangles in multiple bodies (overlapping surface lists) take
the first matching body id.
"""

from __future__ import annotations

__all__ = ["BodyGroupingParams", "body_grouping"]

import pathlib
from typing import ClassVar, Literal

import numpy as np
from lnas import LnasFormat
from pydantic import ConfigDict

from cfdmod.core.data_source import DataSource
from cfdmod.core.grouping import Grouping
from cfdmod.core.ops import OpParams


class BodyGroupingParams(OpParams):
    """Parameters for :func:`body_grouping`.

    Attributes:
        mesh: Path to a ``.lnas`` file.
        bodies: Dict mapping body name -> list of surface names. An
            empty list means "every surface in the mesh".
        name: Grouping name to attach. Defaults to ``"body"``.
    """

    model_config = ConfigDict(frozen=True, arbitrary_types_allowed=True)

    kind: Literal["body_grouping"] = "body_grouping"
    mesh: str
    bodies: dict[str, list[str]]
    name: str = "body"

    chunkable_along: ClassVar[frozenset[str]] = frozenset({"time"})


def body_grouping(ds: DataSource, p: BodyGroupingParams) -> DataSource:
    lnas = LnasFormat.from_file(pathlib.Path(p.mesh))
    if lnas.geometry.triangles.shape[0] != ds.n_elements:
        raise ValueError(
            f"mesh has {lnas.geometry.triangles.shape[0]} triangles but data source has "
            f"{ds.n_elements} elements."
        )

    indices = np.full(ds.n_elements, -1, dtype=np.int32)
    id_to_label: dict[int, str] = {}

    for i, (body_name, surfaces) in enumerate(p.bodies.items()):
        sfcs = surfaces if surfaces else list(lnas.surfaces.keys())
        _, geom_idx = lnas.geometry_from_list_surfaces(surfaces_names=sfcs)
        unassigned = indices[geom_idx] == -1
        indices[geom_idx[unassigned]] = i
        id_to_label[i] = body_name

    grouping = Grouping(name=p.name, indices=indices, id_to_label=id_to_label)
    return ds.with_grouping(grouping)
