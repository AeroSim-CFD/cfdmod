"""Tests for ``cfdmod.remesh.functions``.

Covers:

- ``merge_coplanar`` on a flat NxN-subdivided square (-> 2 triangles), a
  curved patch (unchanged), an L-shape (minimum triangulation of the
  hexagonal boundary), and two separate coplanar regions joined by a
  non-coplanar bridge (each region collapses independently).
- ``decimate_qem`` flat-quad reduction and boundary preservation.
- ``remesh_per_group`` defaults end-to-end on a two-surface synthetic mesh
  and the existing ``regroup`` galpao fixture.
"""

from __future__ import annotations

import pathlib

import numpy as np
import pytest
from lnas import LnasFormat, LnasGeometry
from lnas import fmt as _lnas_fmt

from cfdmod.remesh import decimate_qem, merge_coplanar, remesh_per_group

REPO_ROOT = pathlib.Path(__file__).resolve().parents[2]


def _subdivided_square(n: int, z: float = 0.0) -> tuple[np.ndarray, np.ndarray]:
    """Flat unit square subdivided into ``n x n`` cells, two triangles per cell.

    Returns ``(vertices, triangles)`` with shape ``((n+1)^2, 3)`` and
    ``(2 n^2, 3)`` respectively. All triangles share the same normal (+Z).
    """
    side = n + 1
    xs = np.linspace(0.0, 1.0, side)
    ys = np.linspace(0.0, 1.0, side)
    xv, yv = np.meshgrid(xs, ys, indexing="xy")
    vertices = np.stack(
        [xv.flatten(), yv.flatten(), np.full(xv.size, z)],
        axis=1,
    ).astype(np.float64)
    tris = []
    for j in range(n):
        for i in range(n):
            v0 = j * side + i
            v1 = j * side + (i + 1)
            v2 = (j + 1) * side + i
            v3 = (j + 1) * side + (i + 1)
            # Both winding orders set so that the cross product points to +Z.
            tris.append([v0, v1, v2])
            tris.append([v1, v3, v2])
    return vertices, np.array(tris, dtype=np.int64)


def _curved_patch(n: int) -> tuple[np.ndarray, np.ndarray]:
    """``n x n`` patch of a paraboloid ``z = x^2 + y^2`` on ``[-1, 1]^2``.

    No two adjacent triangles share a plane, so ``merge_coplanar`` is a no-op.
    """
    side = n + 1
    xs = np.linspace(-1.0, 1.0, side)
    ys = np.linspace(-1.0, 1.0, side)
    xv, yv = np.meshgrid(xs, ys, indexing="xy")
    zs = xv**2 + yv**2
    vertices = np.stack([xv.flatten(), yv.flatten(), zs.flatten()], axis=1).astype(np.float64)
    tris = []
    for j in range(n):
        for i in range(n):
            v0 = j * side + i
            v1 = j * side + (i + 1)
            v2 = (j + 1) * side + i
            v3 = (j + 1) * side + (i + 1)
            tris.append([v0, v1, v2])
            tris.append([v1, v3, v2])
    return vertices, np.array(tris, dtype=np.int64)


def _l_shape() -> tuple[np.ndarray, np.ndarray]:
    """Flat L-shape (in the XY plane) made of 3 unit cells (6 triangles).

    Cells: (0,0)-(1,1), (1,0)-(2,1), (0,1)-(1,2). Boundary is a 6-vertex
    concave hexagon; minimum triangulation has 4 triangles.
    """
    # Vertex layout (z=0):
    #
    # 7 - 6
    # |   |
    # 5 - 4 - 3
    # |   |   |
    # 0 - 1 - 2
    vertices = np.array(
        [
            [0.0, 0.0, 0.0],  # 0
            [1.0, 0.0, 0.0],  # 1
            [2.0, 0.0, 0.0],  # 2
            [2.0, 1.0, 0.0],  # 3
            [1.0, 1.0, 0.0],  # 4
            [0.0, 1.0, 0.0],  # 5
            [1.0, 2.0, 0.0],  # 6
            [0.0, 2.0, 0.0],  # 7
        ],
        dtype=np.float64,
    )
    tris = np.array(
        [
            # bottom-left cell
            [0, 1, 5],
            [1, 4, 5],
            # bottom-right cell
            [1, 2, 4],
            [2, 3, 4],
            # top-left cell
            [5, 4, 7],
            [4, 6, 7],
        ],
        dtype=np.int64,
    )
    return vertices, tris


def _two_coplanar_components_bridge() -> tuple[np.ndarray, np.ndarray]:
    """Two coplanar 2x2 squares (in z=0) connected by a non-coplanar bridge
    triangle (one vertex offset in z). Coplanar merge should treat each
    square as its own component and leave the bridge alone.
    """
    # Square A: corners (0,0,0)..(1,1,0); four triangles via center vertex (0.5, 0.5, 0).
    # Square B: corners (2,0,0)..(3,1,0); four triangles via center vertex (2.5, 0.5, 0).
    # Bridge: a triangle linking (1, 0.5, 0) - (2, 0.5, 0) - (1.5, 0.5, 1) (out-of-plane).
    vertices = np.array(
        [
            # Square A: 0-3 corners, 4 center
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [1.0, 1.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.5, 0.5, 0.0],
            # Square B: 5-8 corners, 9 center
            [2.0, 0.0, 0.0],
            [3.0, 0.0, 0.0],
            [3.0, 1.0, 0.0],
            [2.0, 1.0, 0.0],
            [2.5, 0.5, 0.0],
            # Bridge: a (10), b (11), c-out-of-plane (12)
            [1.0, 0.5, 0.0],
            [2.0, 0.5, 0.0],
            [1.5, 0.5, 1.0],
        ],
        dtype=np.float64,
    )
    tris = np.array(
        [
            # Square A fan: CCW around the center, looking from +z.
            [0, 1, 4],
            [1, 2, 4],
            [2, 3, 4],
            [3, 0, 4],
            # Square B fan
            [5, 6, 9],
            [6, 7, 9],
            [7, 8, 9],
            [8, 5, 9],
            # Out-of-plane bridge
            [10, 11, 12],
        ],
        dtype=np.int64,
    )
    return vertices, tris


@pytest.mark.unit
def test_merge_coplanar_flat_square_collapses_to_two_triangles():
    v, t = _subdivided_square(n=10)
    assert t.shape[0] == 200  # sanity: 10x10 cells, 2 triangles each
    new_v, new_t = merge_coplanar(v, t)
    assert new_t.shape[0] == 2
    # The four corners must survive (they are boundary vertices).
    corners = {(0.0, 0.0, 0.0), (1.0, 0.0, 0.0), (1.0, 1.0, 0.0), (0.0, 1.0, 0.0)}
    out_set = {tuple(p) for p in new_v.tolist()}
    assert corners.issubset(out_set)
    # Total area is preserved (1.0 unit square).
    a = new_v[new_t[:, 0]]
    b = new_v[new_t[:, 1]]
    c = new_v[new_t[:, 2]]
    areas = 0.5 * np.linalg.norm(np.cross(b - a, c - a), axis=1)
    assert np.isclose(float(areas.sum()), 1.0)


@pytest.mark.unit
def test_merge_coplanar_curved_patch_is_noop():
    v, t = _curved_patch(n=6)
    new_v, new_t = merge_coplanar(v, t)
    assert new_t.shape[0] == t.shape[0]
    assert new_v.shape[0] == v.shape[0]


@pytest.mark.unit
def test_merge_coplanar_l_shape_minimum_triangulation():
    v, t = _l_shape()
    new_v, new_t = merge_coplanar(v, t)
    # 6-vertex concave polygon -> 4 triangles (n - 2 for a simple polygon).
    assert new_t.shape[0] == 4
    # All six boundary vertices kept.
    assert new_v.shape[0] == 6
    # Area preserved (3 unit squares).
    a = new_v[new_t[:, 0]]
    b = new_v[new_t[:, 1]]
    c = new_v[new_t[:, 2]]
    areas = 0.5 * np.linalg.norm(np.cross(b - a, c - a), axis=1)
    assert np.isclose(float(areas.sum()), 3.0)


@pytest.mark.unit
def test_merge_coplanar_two_components_collapse_independently():
    v, t = _two_coplanar_components_bridge()
    new_v, new_t = merge_coplanar(v, t)
    # Each coplanar square (4-corner boundary) -> 2 triangles. Bridge stays. Total 5.
    assert new_t.shape[0] == 5
    # Bridge triangle out-of-plane vertex must survive.
    out_set = {tuple(p) for p in new_v.tolist()}
    assert (1.5, 0.5, 1.0) in out_set


@pytest.mark.unit
def test_merge_coplanar_empty_input():
    v = np.zeros((0, 3), dtype=np.float64)
    t = np.zeros((0, 3), dtype=np.int64)
    new_v, new_t = merge_coplanar(v, t)
    assert new_t.shape == (0, 3)
    assert new_v.shape == (0, 3)


@pytest.mark.unit
def test_decimate_qem_reduces_triangle_count_and_stays_in_bbox():
    """``decimate_qem`` is intended for curved patches (the user's primary path
    for flat groups is ``merge_coplanar``). What we assert here: the call
    actually reduces the triangle count and the output mesh stays inside the
    input bounding box. fast-simplification does shrink flat boundaries (no
    QEM cost penalty along the plane), so we do not assert exact area.
    """
    pytest.importorskip("fast_simplification")
    v, t = _subdivided_square(n=10)
    new_v, new_t = decimate_qem(v, t, target_reduction=0.5)
    assert new_t.shape[0] < t.shape[0]
    eps = 1e-6
    assert new_v[:, 0].min() >= -eps and new_v[:, 0].max() <= 1.0 + eps
    assert new_v[:, 1].min() >= -eps and new_v[:, 1].max() <= 1.0 + eps


@pytest.mark.unit
def test_decimate_qem_zero_reduction_is_noop():
    pytest.importorskip("fast_simplification")
    v, t = _subdivided_square(n=4)
    new_v, new_t = decimate_qem(v, t, target_reduction=0.0)
    assert new_t.shape[0] == t.shape[0]
    assert new_v.shape[0] == v.shape[0]


@pytest.mark.unit
def test_decimate_qem_without_fast_simplification_raises_helpful_error(monkeypatch):
    import builtins

    real_import = builtins.__import__

    def fake_import(name, *args, **kwargs):
        if name == "fast_simplification":
            raise ImportError("simulated missing dep")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", fake_import)
    v, t = _subdivided_square(n=2)
    with pytest.raises(ImportError, match="aerosim-cfdmod\\[remesh\\]"):
        decimate_qem(v, t, target_reduction=0.5)


def _two_surface_flat_mesh() -> LnasFormat:
    """Two disjoint subdivided unit squares as two named surfaces."""
    v_a, t_a = _subdivided_square(n=8)
    v_b, t_b = _subdivided_square(n=4)
    v_b = v_b.copy()
    v_b[:, 0] += 3.0  # shift square B away from A so they don't overlap
    n_a_v = v_a.shape[0]
    n_a_t = t_a.shape[0]
    n_b_t = t_b.shape[0]
    vertices = np.concatenate([v_a, v_b], axis=0).astype(np.float64)
    triangles = np.concatenate([t_a, t_b + n_a_v], axis=0).astype(np.uint32)
    surfaces = {
        "left": np.arange(0, n_a_t, dtype=np.uint32),
        "right": np.arange(n_a_t, n_a_t + n_b_t, dtype=np.uint32),
    }
    return LnasFormat(
        version=_lnas_fmt._CURRENT_VERSION,
        geometry=LnasGeometry(vertices=vertices, triangles=triangles),
        surfaces=surfaces,
    )


@pytest.mark.unit
def test_remesh_per_group_defaults_collapse_each_surface():
    mesh = _two_surface_flat_mesh()
    n_in = mesh.geometry.triangles.shape[0]
    assert n_in == 2 * 64 + 2 * 16

    out = remesh_per_group(mesh)
    # Each surface (a flat subdivided square) -> 2 triangles. Total: 4.
    assert out.geometry.triangles.shape[0] == 4
    assert list(out.surfaces.keys()) == ["left", "right"]
    assert out.surfaces["left"].size == 2
    assert out.surfaces["right"].size == 2

    # Surface areas preserved.
    g = out.geometry
    a = g.vertices[g.triangles[:, 0]]
    b = g.vertices[g.triangles[:, 1]]
    c = g.vertices[g.triangles[:, 2]]
    areas = 0.5 * np.linalg.norm(np.cross(b - a, c - a), axis=1)
    assert np.isclose(float(areas.sum()), 2.0)


@pytest.mark.unit
def test_remesh_per_group_curved_input_unchanged_with_defaults():
    v, t = _curved_patch(n=5)
    mesh = LnasFormat(
        version=_lnas_fmt._CURRENT_VERSION,
        geometry=LnasGeometry(vertices=v.astype(np.float64), triangles=t.astype(np.uint32)),
        surfaces={"all": np.arange(t.shape[0], dtype=np.uint32)},
    )
    out = remesh_per_group(mesh)
    assert out.geometry.triangles.shape[0] == t.shape[0]


@pytest.mark.unit
def test_remesh_per_group_qem_path_reduces_curved_surface():
    pytest.importorskip("fast_simplification")
    v, t = _curved_patch(n=10)
    mesh = LnasFormat(
        version=_lnas_fmt._CURRENT_VERSION,
        geometry=LnasGeometry(vertices=v.astype(np.float64), triangles=t.astype(np.uint32)),
        surfaces={"all": np.arange(t.shape[0], dtype=np.uint32)},
    )
    n_in = t.shape[0]
    out = remesh_per_group(mesh, target_reduction=0.5)
    assert out.geometry.triangles.shape[0] < n_in


@pytest.mark.unit
def test_remesh_per_group_empty_surface_preserved():
    mesh = _two_surface_flat_mesh()
    # Replace 'right' with an empty surface
    mesh = LnasFormat(
        version=mesh.version,
        geometry=mesh.geometry,
        surfaces={
            "left": mesh.surfaces["left"],
            "right": np.zeros(0, dtype=np.uint32),
        },
    )
    out = remesh_per_group(mesh)
    assert "right" in out.surfaces
    assert out.surfaces["right"].size == 0


# ---------------------------------------------------------------------------
# Hardening regressions (covers fixes from the PR-136 review)
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_merge_coplanar_handles_flipped_winding_in_one_triangle():
    """A coplanar fan with one triangle flipped (anti-parallel normal) should
    still be detected as one coplanar component and collapse to 2 triangles.

    Without anti-parallel handling, the flipped triangle's normal fails the
    same-direction cosine test and the fan splits into two components, leaving
    the mesh effectively un-merged.
    """
    v, t = _subdivided_square(n=4)
    # Flip the winding of one interior triangle (index 5; arbitrary).
    t_mixed = t.copy()
    flip_idx = 5
    t_mixed[flip_idx] = t_mixed[flip_idx][::-1]
    new_v, new_t = merge_coplanar(v, t_mixed)
    # With anti-parallel handling, the whole fan collapses to 2 triangles.
    # Without it, you'd see closer to t.shape[0] - 1 because of the orphaned
    # flipped triangle; assert tightly so a regression is loud.
    assert new_t.shape[0] == 2


@pytest.mark.unit
def test_merge_coplanar_degenerate_triangles_are_kept_as_is():
    """Zero-area input triangles are marked invalid in ``_triangle_planes``
    and skipped at the coplanar-union step, so they survive untouched in
    the output (no crash, no merge).
    """
    v = np.array(
        [
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [2.0, 0.0, 0.0],
            [3.0, 0.0, 0.0],
        ],
        dtype=np.float64,
    )
    # All three vertices of each triangle are collinear -> zero area.
    t = np.array([[0, 1, 2], [1, 2, 3]], dtype=np.int64)
    new_v, new_t = merge_coplanar(v, t)
    assert new_t.shape[0] == t.shape[0]


@pytest.mark.unit
def test_merge_coplanar_branching_boundary_falls_back_with_log(caplog):
    """A coplanar component whose boundary has a branching vertex (more than
    two boundary neighbours) cannot be walked as a simple loop. The function
    must fall back and emit the ``malformed or closed boundary`` debug log.
    """
    import logging

    # Three coplanar triangles all sharing the non-manifold edge (0, 1) on z=0.
    # That edge has count=3 in the undirected map (not boundary), but vertices
    # 0 and 1 each carry three boundary half-edges out to {2, 3, 4}. So both
    # have degree 3 in the boundary graph and the 2-neighbour invariant fails.
    v = np.array(
        [
            [0.0, 0.0, 0.0],  # 0
            [1.0, 0.0, 0.0],  # 1
            [-1.0, 1.0, 0.0],  # 2
            [-1.0, -1.0, 0.0],  # 3
            [2.0, 1.0, 0.0],  # 4
        ],
        dtype=np.float64,
    )
    t = np.array(
        [
            [0, 1, 2],
            [0, 3, 1],
            [0, 1, 4],
        ],
        dtype=np.int64,
    )
    with caplog.at_level(logging.DEBUG, logger="cfdmod"):
        new_v, new_t = merge_coplanar(v, t)
    # Fallback: original triangle count preserved.
    assert new_t.shape[0] == t.shape[0]
    # Log message names the function and the fallback reason.
    assert any(
        "merge_coplanar" in rec.message and "boundary" in rec.message for rec in caplog.records
    )


@pytest.mark.unit
def test_decimate_qem_warns_on_closed_surface():
    """A closed sub-mesh (no boundary edges) emits a RuntimeWarning so the
    caller knows QEM has nothing protecting it from over-collapse.
    """
    pytest.importorskip("fast_simplification")
    # Tetrahedron: 4 vertices, 4 triangles, no boundary edges.
    v = np.array(
        [
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
        ],
        dtype=np.float64,
    )
    t = np.array(
        [
            [0, 2, 1],
            [0, 1, 3],
            [0, 3, 2],
            [1, 2, 3],
        ],
        dtype=np.int32,
    )
    with pytest.warns(RuntimeWarning, match="closed surface"):
        decimate_qem(v, t, target_reduction=0.5)


@pytest.mark.unit
def test_merge_coplanar_multi_loop_component_falls_back_with_log(caplog):
    """An annular coplanar component (square with a square hole, all coplanar)
    cannot be ear-clipped as a single loop; the function should keep the
    original triangles and emit a debug-level log.
    """
    import logging

    # Outer square corners (CCW), inner square hole corners (CW from outer pov).
    v = np.array(
        [
            # outer
            [0.0, 0.0, 0.0],  # 0
            [3.0, 0.0, 0.0],  # 1
            [3.0, 3.0, 0.0],  # 2
            [0.0, 3.0, 0.0],  # 3
            # inner hole
            [1.0, 1.0, 0.0],  # 4
            [2.0, 1.0, 0.0],  # 5
            [2.0, 2.0, 0.0],  # 6
            [1.0, 2.0, 0.0],  # 7
        ],
        dtype=np.float64,
    )
    # 8 triangles spanning the annulus, all in the same plane.
    t = np.array(
        [
            [0, 1, 4],
            [1, 5, 4],
            [1, 2, 5],
            [2, 6, 5],
            [2, 3, 6],
            [3, 7, 6],
            [3, 0, 7],
            [0, 4, 7],
        ],
        dtype=np.int64,
    )
    with caplog.at_level(logging.DEBUG, logger="cfdmod"):
        new_v, new_t = merge_coplanar(v, t)
    # Fallback: triangle count unchanged.
    assert new_t.shape[0] == t.shape[0]
    # At least one fallback log line was emitted.
    assert any("merge_coplanar" in rec.message for rec in caplog.records)


@pytest.mark.unit
def test_remesh_per_group_seam_dedup_tolerance_merges_near_identical_vertices():
    """With ``seam_rel_tol > 0`` (default), two surfaces that share a boundary
    vertex up to sub-float drift end up with that vertex *merged* in the output.
    """
    # Two adjacent flat quads: 'left' covers x in [0, 1], 'right' covers
    # x in [1, 2]. They share the edge at x=1 but with one of the shared
    # vertex coords perturbed by a tiny amount on the 'right' side.
    v_left = np.array(
        [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [1.0, 1.0, 0.0], [0.0, 1.0, 0.0]],
        dtype=np.float64,
    )
    t_left = np.array([[0, 1, 2], [0, 2, 3]], dtype=np.int64)
    eps = 1e-14  # well below the default seam_rel_tol * bbox_diag
    v_right = np.array(
        [
            [1.0 + eps, 0.0, 0.0],
            [2.0, 0.0, 0.0],
            [2.0, 1.0, 0.0],
            [1.0 + eps, 1.0, 0.0],
        ],
        dtype=np.float64,
    )
    t_right = np.array([[0, 1, 2], [0, 2, 3]], dtype=np.int64)
    vertices = np.concatenate([v_left, v_right], axis=0)
    triangles = np.concatenate([t_left, t_right + v_left.shape[0]], axis=0).astype(np.uint32)
    mesh = LnasFormat(
        version=_lnas_fmt._CURRENT_VERSION,
        geometry=LnasGeometry(vertices=vertices, triangles=triangles),
        surfaces={
            "left": np.array([0, 1], dtype=np.uint32),
            "right": np.array([2, 3], dtype=np.uint32),
        },
    )
    out = remesh_per_group(mesh)
    # 4 quad corners + 2 inner-seam corners (both quads collapse to 2 tris each,
    # bordering at the seam). With seam dedup the inner seam vertices are
    # shared: 4 outer corners + 2 seam vertices = 6.
    assert out.geometry.vertices.shape[0] == 6
