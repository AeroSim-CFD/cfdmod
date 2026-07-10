"""Tests for ``cfdmod.regroup.functions``."""

from __future__ import annotations

import h5py
import numpy as np
import pytest
from lnas import LnasFormat

from cfdmod.geometry.grouping import (
    ByConnectivityGrouping,
    ByZoningGrouping,
)
from cfdmod.io.geometry.transformation_config import TransformationConfig
from cfdmod.regroup.functions import (
    apply_regroup_to_timeseries,
    build_regroup_mapping,
    build_regrouped_mesh,
)
from tests.regroup.conftest import make_synthetic_cp_h5


def _zoning_two_x_cells(small_mesh: LnasFormat):
    """Cuts the small_mesh in half along x at x=1.0."""
    return [
        ByZoningGrouping(
            x_intervals=[0.0, 1.0, 2.0001],
            name_template="r{idx}",
        )
    ]


def test_per_triangle_reorders_columns(small_mesh, tmp_path):
    chain = _zoning_two_x_cells(small_mesh)
    grouping = build_regroup_mapping(small_mesh, chain, transformation=None)
    new_lnas, idx = build_regrouped_mesh(
        small_mesh,
        grouping,
        aggregation="per_triangle",
        unassigned_policy="drop",
    )
    # 2x2 grid -> 8 triangles (two per cell), all inside, four per x-cell.
    assert new_lnas.geometry.triangles.shape[0] == 8
    assert sorted(new_lnas.surfaces.keys()) == ["r0", "r1"]

    in_h5 = tmp_path / "in.h5"
    out_h5 = tmp_path / "out.h5"
    data, _times = make_synthetic_cp_h5(in_h5, n_triangles=8, n_steps=3)

    apply_regroup_to_timeseries(
        in_h5,
        out_h5,
        group="cp",
        regroup_index=idx,
        new_triangles=new_lnas.geometry.triangles,
        new_vertices=new_lnas.geometry.vertices,
    )

    with h5py.File(out_h5, "r") as f:
        out_t0 = f["cp"][f"t{0.0}"][:]
    expected = data[0, idx.new_to_parent]
    np.testing.assert_allclose(out_t0, expected)


def test_area_weighted_mean_aggregates_per_group(small_mesh, tmp_path):
    chain = _zoning_two_x_cells(small_mesh)
    grouping = build_regroup_mapping(small_mesh, chain, transformation=None)
    new_lnas, idx = build_regrouped_mesh(
        small_mesh,
        grouping,
        aggregation="area_weighted_mean",
        unassigned_policy="drop",
    )

    in_h5 = tmp_path / "in.h5"
    out_h5 = tmp_path / "out.h5"
    data, _times = make_synthetic_cp_h5(in_h5, n_triangles=8, n_steps=2, seed=42)

    apply_regroup_to_timeseries(
        in_h5,
        out_h5,
        group="cp",
        regroup_index=idx,
        new_triangles=new_lnas.geometry.triangles,
        new_vertices=new_lnas.geometry.vertices,
    )

    with h5py.File(out_h5, "r") as f:
        out_t0 = f["cp"][f"t{0.0}"][:]

    # All triangles are unit-area squares, so weights are uniform 1/N within each group.
    for gi, group_name in enumerate(idx.output_group_names):
        parents = idx.group_parents[gi]
        weights = idx.group_weights[gi]
        np.testing.assert_allclose(weights.sum(), 1.0)
        expected_value = float(np.sum(weights * data[0, parents]))
        # All output triangles in this group share that value.
        in_group = idx.triangle_group_idx == gi
        np.testing.assert_allclose(out_t0[in_group], expected_value)


def test_per_triangle_rejects_overlap(small_mesh):
    # Two specs that both assign the same triangles -> overlap.
    chain = [
        ByZoningGrouping(x_intervals=[0.0, 2.001], name_template="full_a"),
        ByZoningGrouping(x_intervals=[0.0, 2.001], name_template="full_b"),
    ]
    grouping = build_regroup_mapping(small_mesh, chain, transformation=None)
    with pytest.raises(ValueError, match="per_triangle aggregation requires"):
        build_regrouped_mesh(
            small_mesh,
            grouping,
            aggregation="per_triangle",
            unassigned_policy="drop",
        )


def test_unassigned_kept_when_requested(small_mesh):
    # Zoning that only covers x in [0, 1) -> half the triangles unassigned.
    chain = [
        ByZoningGrouping(
            x_intervals=[0.0, 1.0],
            name_template="r{idx}",
        )
    ]
    grouping = build_regroup_mapping(small_mesh, chain, transformation=None)
    _new_lnas, idx_drop = build_regrouped_mesh(
        small_mesh,
        grouping,
        aggregation="per_triangle",
        unassigned_policy="drop",
    )
    new_lnas_keep, idx_keep = build_regrouped_mesh(
        small_mesh,
        grouping,
        aggregation="per_triangle",
        unassigned_policy="keep_as_unassigned",
    )
    # 2x2 grid = 8 triangles total. Half inside the x in [0,1) cell.
    assert idx_drop.new_to_parent.size == 4
    assert idx_keep.new_to_parent.size == 8
    assert "unassigned" in new_lnas_keep.surfaces


def test_transformation_moves_binning_frame(small_mesh):
    """Rotating by 90deg around z should swap which triangles fall into x cells."""
    # Without transformation: x in [0,1) vs [1,2).
    chain = [
        ByZoningGrouping(
            x_intervals=[0.0, 1.0, 2.001],
            name_template="r{idx}",
        )
    ]
    g_world = build_regroup_mapping(small_mesh, chain, transformation=None)

    # With a 90deg rotation (around z, fixed_point=center), the original Y
    # coordinates become the new X coordinates -- so the bin assignment changes.
    rot = TransformationConfig(
        translation=(0.0, 0.0, 0.0),
        rotation=(0.0, 0.0, np.pi / 2),
        fixed_point=(1.0, 1.0, 0.0),
    )
    g_rot = build_regroup_mapping(small_mesh, chain, transformation=rot)

    # The two groupings must produce different triangle assignments.
    world_r0 = set(int(i) for i in g_world.groups.get("r0", []))
    rot_r0 = set(int(i) for i in g_rot.groups.get("r0", []))
    assert world_r0 != rot_r0


def test_slice_triangles_with_parents_no_planes_passthrough(small_mesh):
    from cfdmod.regroup.functions import slice_triangles_with_parents

    n = small_mesh.geometry.triangles.shape[0]
    parent_idxs = np.arange(n, dtype=np.int64)
    intervals = (
        [float("-inf"), float("inf")],
        [float("-inf"), float("inf")],
        [float("-inf"), float("inf")],
    )
    verts, normals, parents = slice_triangles_with_parents(
        small_mesh.geometry.triangle_vertices,
        small_mesh.geometry.normals,
        parent_idxs,
        intervals,
    )
    # No finite cut planes -> no slicing happens; identity output.
    assert verts.shape == (n, 3, 3)
    assert parents.tolist() == parent_idxs.tolist()


def test_slice_triangles_with_parents_actually_cuts():
    """A single z-plane triangle straddling x=1.0 must split into 2 sub-tris."""
    from cfdmod.regroup.functions import slice_triangles_with_parents

    # One triangle in the XY plane spanning x=[0,2], y=[0,1], normal +z.
    tri = np.array([[[0.0, 0.0, 0.0], [2.0, 0.0, 0.0], [1.0, 1.0, 0.0]]])
    normals = np.array([[0.0, 0.0, 1.0]])
    parents = np.array([42], dtype=np.int64)
    intervals = (
        [float("-inf"), 1.0, float("inf")],
        [float("-inf"), float("inf")],
        [float("-inf"), float("inf")],
    )
    verts, _normals, parents_out = slice_triangles_with_parents(tri, normals, parents, intervals)
    # The triangle straddles the x=1 plane, so it slices.
    assert verts.shape[0] >= 2
    # Every fragment retains the parent index.
    assert np.all(parents_out == 42)


def _slice_triangles_naive(
    tri_verts: np.ndarray,
    tri_normals: np.ndarray,
    parent_idxs: np.ndarray,
    intervals,
):
    """Reference per-fragment slicer (the pre-vectorisation implementation).

    Built directly on ``_slice_one_triangle`` so the vectorised
    ``slice_triangles_with_parents`` can be checked for exact parity.
    """
    from cfdmod.regroup.functions import _slice_one_triangle

    cur_verts = tri_verts.astype(np.float64).copy()
    cur_normals = tri_normals.astype(np.float64).copy()
    cur_parents = np.asarray(parent_idxs, dtype=np.int64).copy()
    for axis in range(3):
        for v in intervals[axis]:
            if not np.isfinite(v):
                continue
            new_verts, new_normals, new_parents = [], [], []
            for i in range(cur_verts.shape[0]):
                fragments = _slice_one_triangle(cur_verts[i], cur_normals[i], axis, float(v))
                new_verts.append(fragments)
                new_normals.append(np.tile(cur_normals[i], (fragments.shape[0], 1)))
                new_parents.append(np.full(fragments.shape[0], cur_parents[i], dtype=np.int64))
            cur_verts = np.concatenate(new_verts, axis=0)
            cur_normals = np.concatenate(new_normals, axis=0)
            cur_parents = np.concatenate(new_parents, axis=0)
    return cur_verts, cur_normals, cur_parents


def test_slice_vectorised_matches_naive_on_curved_mesh(curved_mesh):
    """Vectorised slicer is bit-identical to the naive per-fragment loop.

    The curved fixture straddles all three axes, so this exercises the
    slicing path the planar fixtures mostly skip.
    """
    from cfdmod.regroup.functions import slice_triangles_with_parents

    n = curved_mesh.geometry.triangles.shape[0]
    parent_idxs = np.arange(n, dtype=np.int64)
    verts = curved_mesh.geometry.triangle_vertices
    normals = curved_mesh.geometry.normals
    intervals = (
        [float("-inf"), 3.5, 6.5, 9.5, float("inf")],
        [float("-inf"), 4.5, 8.5, float("inf")],
        [float("-inf"), -0.2, 0.2, float("inf")],
    )

    v_new, n_new, p_new = slice_triangles_with_parents(verts, normals, parent_idxs, intervals)
    v_ref, n_ref, p_ref = _slice_triangles_naive(verts, normals, parent_idxs, intervals)

    assert v_new.shape[0] > n  # slicing actually happened
    np.testing.assert_array_equal(v_new, v_ref)
    np.testing.assert_array_equal(n_new, n_ref)
    np.testing.assert_array_equal(p_new, p_ref)


def test_slice_partial_areas_sum_to_parent(curved_mesh):
    """Fragments of a cut triangle partition its area (exact partial areas)."""
    from cfdmod.regroup.functions import slice_triangles_with_parents

    n = curved_mesh.geometry.triangles.shape[0]
    parent_idxs = np.arange(n, dtype=np.int64)
    verts = curved_mesh.geometry.triangle_vertices
    normals = curved_mesh.geometry.normals
    intervals = (
        [float("-inf"), 3.5, 7.5, float("inf")],
        [float("-inf"), 5.5, float("inf")],
        [float("-inf"), float("inf")],
    )

    frag_verts, _frag_normals, frag_parents = slice_triangles_with_parents(
        verts, normals, parent_idxs, intervals
    )

    def _tri_area(t):
        return 0.5 * np.linalg.norm(np.cross(t[:, 1] - t[:, 0], t[:, 2] - t[:, 0]), axis=-1)

    parent_area = _tri_area(verts.astype(np.float64))
    frag_area = _tri_area(frag_verts)
    summed = np.zeros(n, dtype=np.float64)
    np.add.at(summed, frag_parents, frag_area)
    np.testing.assert_allclose(summed, parent_area, rtol=1e-5, atol=1e-6)


def test_two_container_connectivity_split(two_container_mesh):
    """Connectivity isolates the two containers as separate groups."""
    chain = [ByConnectivityGrouping(name_template="container_{idx}", min_triangles=4)]
    grouping = build_regroup_mapping(two_container_mesh, chain, transformation=None)
    assert len(grouping.groups) == 2
    sizes = sorted(int(idxs.size) for idxs in grouping.groups.values())
    # Container A: 4*6*2 = 48 triangles; Container B: 2*3*2 = 12 triangles.
    assert sizes == [12, 48]
