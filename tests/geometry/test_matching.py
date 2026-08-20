"""Unit tests for reference-mesh triangle matching."""

from __future__ import annotations

import pathlib

import numpy as np
import pytest

from cfdmod.geometry.matching import MISSING, as_triangle_vertices, match_triangles

GALPAO = pathlib.Path("fixtures/tests/pressure/galpao")


def _grid_triangles(n: int, *, offset: float = 0.0) -> np.ndarray:
    """``n`` unit right triangles laid out along x, optionally translated."""
    base = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])
    tris = np.stack([base + np.array([2.0 * i, 0.0, 0.0]) for i in range(n)])
    return tris + offset


def test_identity_match_is_the_identity_permutation():
    ref = _grid_triangles(6)
    m = match_triangles(ref, ref)
    np.testing.assert_array_equal(m.indices, np.arange(6))
    assert m.is_complete and m.is_injective
    assert m.n_missing == 0
    np.testing.assert_array_equal(m.distances, np.zeros(6))


def test_submesh_indices_follow_the_target_order():
    ref = _grid_triangles(6)
    picked = [4, 1, 5]
    m = match_triangles(ref, ref[picked])
    np.testing.assert_array_equal(m.indices, picked)
    assert m.is_complete


def test_vertex_rotation_and_winding_flip_still_match():
    ref = _grid_triangles(4)
    rotated = ref[[2, 0]][:, [1, 2, 0], :]  # rotate vertex order
    flipped = ref[[3]][:, ::-1, :]  # reverse winding
    target = np.concatenate([rotated, flipped])
    np.testing.assert_array_equal(match_triangles(ref, target).indices, [2, 0, 3])


def test_triangle_absent_from_reference_is_flagged_missing():
    ref = _grid_triangles(4)
    stranger = _grid_triangles(1, offset=1000.0)
    target = np.concatenate([ref[[1]], stranger, ref[[3]]])
    m = match_triangles(ref, target)
    np.testing.assert_array_equal(m.indices, [1, MISSING, 3])
    np.testing.assert_array_equal(m.missing, [False, True, False])
    assert m.n_found == 2 and m.n_missing == 1
    assert not m.is_complete
    assert np.isinf(m.distances[1])


def test_missing_is_nan_in_the_float_view():
    ref = _grid_triangles(3)
    target = np.concatenate([ref[[2]], _grid_triangles(1, offset=500.0)])
    as_float = match_triangles(ref, target).as_float()
    assert as_float[0] == 2.0
    assert np.isnan(as_float[1])


def test_require_complete_raises_on_a_missing_triangle():
    ref = _grid_triangles(3)
    target = _grid_triangles(1, offset=500.0)
    m = match_triangles(ref, target)
    with pytest.raises(ValueError, match="no match in the reference geometry"):
        m.require_complete()
    # And returns the indices untouched when everything matched.
    np.testing.assert_array_equal(match_triangles(ref, ref).require_complete(), [0, 1, 2])


def test_repeated_target_triangles_match_the_same_reference_index():
    ref = _grid_triangles(4)
    m = match_triangles(ref, ref[[2, 2]])
    np.testing.assert_array_equal(m.indices, [2, 2])
    assert m.is_complete
    assert not m.is_injective


def test_coincident_reference_triangles_are_reported_ambiguous():
    ref = np.concatenate([_grid_triangles(3), _grid_triangles(3)[[1]]])
    with pytest.warns(RuntimeWarning, match="not clearly below"):
        m = match_triangles(ref, _grid_triangles(3)[[1]])
    assert m.is_complete
    assert bool(m.ambiguous[0])


def test_tolerance_too_tight_rejects_a_perturbed_triangle():
    ref = _grid_triangles(4)
    target = ref[[1]] + 1e-3
    assert match_triangles(ref, target, atol=1e-2).is_complete
    assert match_triangles(ref, target, atol=1e-6).n_missing == 1


def test_area_check_rejects_a_different_triangle_at_the_same_centroid():
    ref = _grid_triangles(3)
    # Same centroid as ref[1], three times the size.
    centroid = ref[1].mean(axis=0)
    scaled = centroid + 3.0 * (ref[1] - centroid)
    assert match_triangles(ref, scaled[None, ...]).n_missing == 1
    # Disabling the cross-check lets centroid proximity alone decide.
    assert match_triangles(ref, scaled[None, ...], area_rtol=None).indices[0] == 1


def test_empty_target_returns_an_empty_match():
    m = match_triangles(_grid_triangles(3), np.empty((0, 3, 3)))
    assert m.indices.shape == (0,)
    assert m.is_complete and m.is_injective


def test_empty_reference_raises():
    with pytest.raises(ValueError, match="no triangles"):
        match_triangles(np.empty((0, 3, 3)), _grid_triangles(1))


def test_bad_geometry_shape_raises():
    with pytest.raises(ValueError, match=r"shape \(n, 3, 3\)"):
        match_triangles(np.zeros((4, 3)), _grid_triangles(1))


def test_unsupported_geometry_file_extension_raises(tmp_path):
    obj = tmp_path / "mesh.obj"
    obj.write_text("")
    with pytest.raises(ValueError, match="unsupported geometry file"):
        match_triangles(obj, _grid_triangles(1))


# ----- Real fixtures: the galpao sub-STLs of the galpao reference mesh --------


def test_as_triangle_vertices_reads_stl_and_lnas():
    from_stl = as_triangle_vertices(GALPAO / "galpao.normalized.stl")
    from_lnas = as_triangle_vertices(GALPAO / "galpao.normalized.lnas")
    assert from_stl.shape == from_lnas.shape == (2915, 3, 3)
    assert from_stl.dtype == np.float64


@pytest.mark.parametrize("sub_stl", ["L1_xp.stl", "L2_yp.stl", "t1_ym.stl", "p6c_yp.stl"])
def test_galpao_sub_stl_matches_the_reference_mesh(sub_stl):
    m = match_triangles(GALPAO / "galpao.lnas", GALPAO / sub_stl)
    n_target = as_triangle_vertices(GALPAO / sub_stl).shape[0]
    assert m.indices.shape == (n_target,)
    assert m.is_complete, f"{sub_stl}: {m.n_missing} unmatched triangles"
    assert m.is_injective
    assert not m.ambiguous.any()
    assert m.distances.max() < m.tolerance


def test_galpao_sub_stls_tile_the_reference_mesh_exactly():
    """The 20 region STLs together cover every reference triangle once."""
    ref = GALPAO / "galpao.lnas"
    covered: list[int] = []
    for stl in sorted(GALPAO.glob("*.stl")):
        if stl.name.startswith("galpao"):
            continue
        covered.extend(match_triangles(ref, stl).require_complete().tolist())
    assert sorted(covered) == list(range(2915))


def test_sub_stl_of_the_wrong_reference_is_all_missing():
    """The region STLs live in unnormalized coordinates -- no false matches."""
    m = match_triangles(GALPAO / "galpao.normalized.lnas", GALPAO / "L2_yp.stl")
    assert m.n_found == 0
    assert np.all(m.as_float() != m.as_float())  # every entry NaN
