"""Geometrically slice a surface's triangles along axis-aligned planes.

``face_cut`` cuts each triangle of a :class:`SurfaceDataSource` along a
set of fixed axis-aligned coordinates and emits a *new* surface whose
fragments each sit cleanly inside one slice. Every fragment inherits its
parent triangle's fields (piecewise-constant: a child copies the
parent's timeseries) and its parent's oriented normal, and recomputes
its own area / centroid.

This is the exact counterpart to the centroid-based
:func:`cfdmod.core.ops.geometric.zoning_grouping`: instead of assigning a
*whole* triangle to one slice by its centroid (an approximation when a
triangle straddles a boundary), ``face_cut`` splits the triangle so each
side contributes its real partial area. Downstream,
``force_contribution`` + ``field_series_for_groups(agg="sum")`` integrate
per slice unchanged, giving exact force / moment by floor or zone.

The cut arithmetic is the shared, dependency-light core in
:mod:`cfdmod.io.geometry.triangle_slicing` (also used by the legacy
``cfdmod.regroup`` pipeline), so this op is an adapter across the storage
seam rather than new geometry math.

Region ids are 0-indexed integers in raster order
(``ix + nx * iy + nx * ny * iz``), matching ``zoning_grouping`` so the two
ops are interchangeable in a template.
"""

from __future__ import annotations

__all__ = ["FaceCutParams", "face_cut"]

from typing import ClassVar, Literal

import numpy as np
from pydantic import ConfigDict

from cfdmod.adapters.memory import MemoryFieldStore
from cfdmod.core.data_source import DataSource, SurfaceDataSource
from cfdmod.core.grouping import Grouping
from cfdmod.core.ops import OpParams
from cfdmod.core.topology import ElementMeta, Topology
from cfdmod.geometry.triangle_slicing import (
    bin_centroid_to_cell,
    build_geometry_from_fragments,
    slice_triangles_with_parents,
)


class FaceCutParams(OpParams):
    """Parameters for :func:`face_cut`.

    Attributes:
        x_intervals: Monotone increasing cut planes / bin edges along x.
            ``n`` edges produce ``n - 1`` bins. Empty list -> open axis
            (no x cut, everything in x-bin 0). For per-floor slicing set
            only ``z_intervals`` and leave x/y open.
        y_intervals: Same convention along y.
        z_intervals: Same convention along z.
        name: Grouping name attached on the output (e.g. ``"floor"``).
        unassigned_policy: ``"drop"`` removes fragments whose centroid
            falls outside every declared bin; ``"keep_as_unassigned"``
            keeps them with group id ``-1``.
    """

    model_config = ConfigDict(frozen=True)

    kind: Literal["face_cut"] = "face_cut"
    x_intervals: list[float] = []
    y_intervals: list[float] = []
    z_intervals: list[float] = []
    name: str = "floor"
    unassigned_policy: Literal["drop", "keep_as_unassigned"] = "drop"

    chunkable_along: ClassVar[frozenset[str]] = frozenset({"time"})
    consumes: ClassVar[frozenset[str] | None] = frozenset({"surface"})
    produces: ClassVar[str] = "surface"
    requires_element_meta: ClassVar[frozenset[str]] = frozenset({"normal"})
    produces_element_meta: ClassVar[frozenset[str]] = frozenset({"area", "normal", "position"})


def _normalize_intervals(edges: list[float]) -> list[float]:
    """Empty -> open infinite interval; otherwise pass through."""
    if not edges:
        return [float("-inf"), float("inf")]
    return edges


def _triangle_areas(tri_verts: np.ndarray) -> np.ndarray:
    """Per-triangle area from ``(m, 3, 3)`` vertex arrays."""
    e1 = tri_verts[:, 1] - tri_verts[:, 0]
    e2 = tri_verts[:, 2] - tri_verts[:, 0]
    return 0.5 * np.linalg.norm(np.cross(e1, e2), axis=1)


def face_cut(ds: DataSource, p: FaceCutParams) -> SurfaceDataSource:
    if ds.topology is None or ds.topology.cell_type != "triangle":
        raise ValueError(
            "face_cut requires a triangle Topology; got "
            f"{None if ds.topology is None else ds.topology.cell_type!r}"
        )
    if ds.elements.normal is None:
        raise ValueError(
            "face_cut requires elements.normal (fragments inherit the parent's "
            "oriented normal); attach it with mesh_attach first."
        )

    n_parents = ds.n_elements
    tri_verts = ds.topology.vertices[ds.topology.connectivity]  # (n, 3, 3)
    tri_normals = np.asarray(ds.elements.normal, dtype=np.float64)
    parent_idxs = np.arange(n_parents, dtype=np.int64)

    intervals = (
        _normalize_intervals(p.x_intervals),
        _normalize_intervals(p.y_intervals),
        _normalize_intervals(p.z_intervals),
    )

    frag_verts, frag_normals, parent_per_fragment = slice_triangles_with_parents(
        tri_verts, tri_normals, parent_idxs, intervals
    )

    # Region id per fragment from its centroid bin, in raster order.
    nx = max(len(intervals[0]) - 1, 1)
    ny = max(len(intervals[1]) - 1, 1)
    centroids = frag_verts.mean(axis=1)
    region_ids = np.empty(frag_verts.shape[0], dtype=np.int32)
    for i in range(frag_verts.shape[0]):
        ix, iy, iz = bin_centroid_to_cell(centroids[i], intervals)
        if ix < 0 or iy < 0 or iz < 0:
            region_ids[i] = -1
        else:
            region_ids[i] = ix + nx * iy + (nx * ny) * iz

    # Apply the unassigned policy before building topology so every array
    # stays row-aligned.
    if p.unassigned_policy == "drop":
        keep = region_ids >= 0
    else:
        keep = np.ones(frag_verts.shape[0], dtype=bool)
    if not keep.any():
        raise ValueError(
            "face_cut produced 0 assigned fragments; check the intervals against the mesh extent."
        )

    frag_verts = frag_verts[keep]
    frag_normals = frag_normals[keep]
    parent_per_fragment = parent_per_fragment[keep]
    region_ids = region_ids[keep]
    centroids = centroids[keep]

    vertices, triangles = build_geometry_from_fragments(frag_verts)
    new_topology = Topology.triangles(triangles, vertices)
    new_elements = ElementMeta(
        position=centroids,
        area=_triangle_areas(frag_verts),
        normal=frag_normals,
    )

    # Fields: gather each fragment's timeseries from its parent row. This
    # repeats a parent's row for each of its fragments (piecewise-constant
    # inheritance). Works for both memory (direct fancy-index) and h5
    # (per-timestep gather) field stores.
    gathered: dict[str, np.ndarray] = {}
    for fname in ds.fields.keys():
        gathered[fname] = ds.fields.read(fname, elements=parent_per_fragment)

    grouping = Grouping(name=p.name, indices=region_ids)

    return SurfaceDataSource(
        time=ds.time,
        topology=new_topology,
        elements=new_elements,
        groupings={p.name: grouping},
        fields=MemoryFieldStore(gathered),
        field_meta={name: meta for name, meta in ds.field_meta.items() if name in gathered},
        attrs=dict(ds.attrs),
    )
