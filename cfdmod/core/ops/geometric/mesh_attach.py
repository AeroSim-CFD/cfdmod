"""Attach mesh-derived element metadata to a SurfaceDataSource.

Takes a path to an ``.lnas`` mesh file and a data source whose
topology has the same triangle count, and returns a new data source
with ``elements.area``, ``elements.normal``, and ``elements.position``
(triangle centroids) populated from the mesh.

Used by every Cf/Cm/Ce recipe: the cp time series comes from disk
without per-triangle areas / normals (the v2 layout never wrote them),
but the force / moment calculations need them.
"""

from __future__ import annotations

__all__ = ["MeshAttachParams", "mesh_attach"]

import pathlib
from typing import ClassVar, Literal

from lnas import LnasFormat
from pydantic import ConfigDict

from cfdmod.core.data_source import DataSource
from cfdmod.core.ops import OpParams
from cfdmod.core.topology import ElementMeta


class MeshAttachParams(OpParams):
    """Parameters for :func:`mesh_attach`.

    Attributes:
        mesh: Path to a ``.lnas`` file. Must produce a triangle count
            matching the data source's ``n_elements``.
    """

    model_config = ConfigDict(frozen=True, arbitrary_types_allowed=True)

    kind: Literal["mesh_attach"] = "mesh_attach"
    mesh: str

    chunkable_along: ClassVar[frozenset[str]] = frozenset({"time"})
    produces_element_meta: ClassVar[frozenset[str]] = frozenset({"area", "normal", "position"})


def mesh_attach(ds: DataSource, p: MeshAttachParams) -> DataSource:
    lnas = LnasFormat.from_file(pathlib.Path(p.mesh))
    geom = lnas.geometry
    if geom.triangles.shape[0] != ds.n_elements:
        raise ValueError(
            f"mesh has {geom.triangles.shape[0]} triangles but data source has "
            f"{ds.n_elements} elements; mesh and cp source must match."
        )

    centroids = geom.triangle_vertices.mean(axis=1)
    elements = ElementMeta(
        position=centroids,
        area=geom.areas,
        normal=geom.normals,
    )
    return ds.with_elements(elements)
