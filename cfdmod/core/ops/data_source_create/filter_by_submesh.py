"""Filter the elements of a data source down to a sub-mesh geometry.

Given a data source whose elements are the triangles of a reference mesh,
and a second geometry that is a sub-mesh of it (typically an ``.stl``
exported for a region of interest), keep only the elements the sub-mesh
covers -- in the sub-mesh's own triangle order, so the result lines up
element-for-element with that file.

Matching is :func:`cfdmod.geometry.matching.match_triangles`: triangle
centroids within a tolerance, cross-checked by area. Sub-mesh triangles
with no counterpart in the reference are an error by default
(``on_missing="error"``) or silently dropped (``on_missing="drop"``); use
:func:`~cfdmod.geometry.matching.match_triangles` directly when you want
the raw index array with its ``-1`` / ``NaN`` sentinels instead of a
filtered data source.
"""

from __future__ import annotations

__all__ = ["FilterBySubmeshParams", "filter_by_submesh"]

from typing import ClassVar, Literal

import numpy as np

from cfdmod.core.data_source import DataSource
from cfdmod.core.ops import OpParams
from cfdmod.core.ops.data_source_create._element_slice import slice_data_source


class FilterBySubmeshParams(OpParams):
    """Parameters for :func:`filter_by_submesh`.

    Attributes:
        mesh: Path to the sub-mesh geometry to extract (``.stl`` or
            ``.lnas``).
        reference_mesh: Path to the reference geometry the data source's
            elements belong to. Optional: when omitted the reference
            triangles are taken from the data source's own triangle
            topology.
        rtol: Relative centroid tolerance, taken against the reference mesh
            length scale. Ignored when ``atol`` is set.
        atol: Absolute centroid tolerance in model units; overrides
            ``rtol``.
        area_rtol: Relative area agreement required of a candidate match.
            ``None`` skips the area cross-check.
        on_missing: ``error`` (default) rejects a sub-mesh triangle with no
            match; ``drop`` keeps only the matched ones.
    """

    kind: Literal["filter_by_submesh"] = "filter_by_submesh"
    mesh: str
    reference_mesh: str | None = None
    rtol: float = 1e-6
    atol: float | None = None
    area_rtol: float | None = 1e-2
    on_missing: Literal["error", "drop"] = "error"

    chunkable_along: ClassVar[frozenset[str]] = frozenset({"time"})


def _reference_triangles(ds: DataSource, p: FilterBySubmeshParams) -> np.ndarray:
    """The reference triangle vertices, from the params path or the topology."""
    from cfdmod.geometry.matching import as_triangle_vertices

    if p.reference_mesh is not None:
        return as_triangle_vertices(p.reference_mesh, "reference_mesh")
    topo = ds.topology
    if topo is None or topo.cell_type != "triangle":
        raise ValueError(
            "filter_by_submesh needs a reference geometry: pass reference_mesh, or "
            "run it on a data source carrying a triangle topology"
        )
    return as_triangle_vertices(topo.vertices[topo.connectivity], "topology")


def filter_by_submesh(ds: DataSource, p: FilterBySubmeshParams) -> DataSource:
    from cfdmod.geometry.matching import match_triangles

    reference = _reference_triangles(ds, p)
    if reference.shape[0] != ds.n_elements:
        raise ValueError(
            f"reference geometry has {reference.shape[0]} triangles but data source "
            f"has {ds.n_elements} elements"
        )

    match = match_triangles(
        reference=reference,
        target=p.mesh,
        rtol=p.rtol,
        atol=p.atol,
        area_rtol=p.area_rtol,
    )

    if p.on_missing == "error":
        keep_idx = match.require_complete()
    else:
        keep_idx = match.indices[match.found]

    if keep_idx.size == 0:
        raise ValueError(
            f"filter_by_submesh selects 0 elements (mesh={p.mesh!r}, "
            f"centroid tolerance {match.tolerance:.3e})"
        )
    return slice_data_source(ds, np.asarray(keep_idx, dtype=np.int64))
