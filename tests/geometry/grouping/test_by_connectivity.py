"""Tests for ByConnectivityGrouping (shared-edge connected components)."""

from __future__ import annotations

import numpy as np
import pytest
from lnas import LnasFormat, LnasGeometry

from cfdmod.geometry import (
    ByConnectivityGrouping,
    BySurfaceGrouping,
    apply_groupings,
)


def _two_disjoint_components() -> LnasFormat:
    """Two independent triangles that share no vertices.

    Component A: triangle 0 with vertex indices {0,1,2}.
    Component B: triangle 1 with vertex indices {3,4,5}.
    """
    vertices = np.array(
        [
            [0, 0, 0],
            [1, 0, 0],
            [0, 1, 0],
            [10, 10, 0],
            [11, 10, 0],
            [10, 11, 0],
        ],
        dtype=np.float32,
    )
    triangles = np.array([[0, 1, 2], [3, 4, 5]], dtype=np.uint32)
    geometry = LnasGeometry(vertices=vertices, triangles=triangles)
    return LnasFormat(
        version="v1.0",
        geometry=geometry,
        surfaces={"all": np.arange(2, dtype=np.uint32)},
    )


def _square_two_triangles() -> LnasFormat:
    """A unit square cut into two triangles sharing one edge.

    Tri 0 = {0,1,2}; Tri 1 = {0,2,3}; shared edge = (0,2).
    """
    vertices = np.array(
        [[0, 0, 0], [1, 0, 0], [1, 1, 0], [0, 1, 0]], dtype=np.float32
    )
    triangles = np.array([[0, 1, 2], [0, 2, 3]], dtype=np.uint32)
    geometry = LnasGeometry(vertices=vertices, triangles=triangles)
    return LnasFormat(
        version="v1.0",
        geometry=geometry,
        surfaces={"sq": np.arange(2, dtype=np.uint32)},
    )


def test_disjoint_components_split():
    mesh = _two_disjoint_components()
    res = apply_groupings(mesh, [ByConnectivityGrouping()])
    # Two components of equal size (1) -> ordering by descending size, then
    # by first triangle index.
    assert set(res.groups) == {"cc0", "cc1"}
    assert res.groups["cc0"].tolist() == [0]
    assert res.groups["cc1"].tolist() == [1]


def test_shared_edge_merges_into_one_component():
    mesh = _square_two_triangles()
    res = apply_groupings(mesh, [ByConnectivityGrouping()])
    assert set(res.groups) == {"cc0"}
    assert sorted(res.groups["cc0"].tolist()) == [0, 1]


def test_min_triangles_filters_small_components():
    # Two components of sizes 2 (square) and 1 (lone triangle).
    sq = _square_two_triangles()
    lone = _two_disjoint_components().geometry  # has 6 verts, 2 disjoint tris

    # Glue: shift the lone-mesh vertices so indices are unique relative to sq,
    # and combine into one LnasFormat.
    n_sq_v = sq.geometry.vertices.shape[0]
    combined_vertices = np.concatenate([sq.geometry.vertices, lone.vertices], axis=0)
    combined_triangles = np.concatenate(
        [sq.geometry.triangles, lone.triangles + n_sq_v], axis=0
    ).astype(np.uint32)
    combined_geometry = LnasGeometry(
        vertices=combined_vertices, triangles=combined_triangles
    )
    combined = LnasFormat(
        version="v1.0",
        geometry=combined_geometry,
        surfaces={"sq": np.array([0, 1], dtype=np.uint32)},
    )

    res = apply_groupings(combined, [ByConnectivityGrouping(min_triangles=2)])
    # Only the size-2 square survives.
    assert len(res.groups) == 1
    assert sorted(next(iter(res.groups.values())).tolist()) == [0, 1]


def test_restrict_to_ignores_outside_edges():
    # Square (sharing an edge) plus a third triangle sharing one edge with
    # the square. Without restrict_to we get a single component of size 3.
    # With restrict_to limited to the original square, we recover the two
    # triangles only (and the third triangle is unrelated, yielding cc1).
    vertices = np.array(
        [[0, 0, 0], [1, 0, 0], [1, 1, 0], [0, 1, 0], [2, 0, 0]],
        dtype=np.float32,
    )
    # Tri 0 = {0,1,2}; Tri 1 = {0,2,3}; Tri 2 = {1,2,4} (shares edge (1,2)
    # with Tri 0).
    triangles = np.array([[0, 1, 2], [0, 2, 3], [1, 2, 4]], dtype=np.uint32)
    geometry = LnasGeometry(vertices=vertices, triangles=triangles)
    mesh = LnasFormat(
        version="v1.0",
        geometry=geometry,
        surfaces={
            "sq": np.array([0, 1], dtype=np.uint32),
            "wing": np.array([2], dtype=np.uint32),
        },
    )

    # No restriction: all three triangles are one component.
    res_full = apply_groupings(mesh, [ByConnectivityGrouping()])
    assert set(res_full.groups) == {"cc0"}
    assert sorted(res_full.groups["cc0"].tolist()) == [0, 1, 2]

    # Restrict to "sq" only: edges to triangle 2 are ignored, so we get
    # one component of size 2.
    res_restricted = apply_groupings(
        mesh,
        [
            BySurfaceGrouping(sets={"sq": ["sq"]}),
            ByConnectivityGrouping(restrict_to=["sq"]),
        ],
    )
    assert "cc0" in res_restricted.groups
    assert sorted(res_restricted.groups["cc0"].tolist()) == [0, 1]
    # Triangle 2 is excluded entirely.
    assert all(2 not in idxs.tolist() for name, idxs in res_restricted.groups.items() if name.startswith("cc"))


def test_name_template_collision_raises():
    mesh = _two_disjoint_components()
    spec = ByConnectivityGrouping(name_template="constant")
    with pytest.raises(ValueError, match="duplicate group name"):
        apply_groupings(mesh, [spec])


def test_grid_mesh_has_one_component_per_triangle(grid_mesh):
    # The grid_mesh fixture assigns unique vertex indices to every
    # triangle (no deduplication), so no triangles share an edge. We
    # therefore expect six singleton components -- this pins the
    # connectivity semantics: it is *vertex-index* connectivity, not
    # geometric coincidence.
    res = apply_groupings(grid_mesh, [ByConnectivityGrouping()])
    assert len(res.groups) == 6
    for idxs in res.groups.values():
        assert idxs.size == 1
