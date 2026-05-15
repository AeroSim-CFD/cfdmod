"""Tests for ``cfdmod.regroup.functions``."""

from __future__ import annotations

import h5py
import numpy as np
import pytest
from lnas import LnasFormat, LnasGeometry, fmt as _lnas_fmt

from cfdmod.geometry.grouping import (
    ByConnectivityGrouping,
    ByZoningGrouping,
    apply_groupings,
)
from cfdmod.io.geometry.transformation_config import TransformationConfig
from cfdmod.regroup.functions import (
    apply_regroup_to_timeseries,
    build_regrouped_mesh,
    build_regroup_mapping,
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
        ByZoningGrouping(
            x_intervals=[0.0, 2.001], name_template="full_a"
        ),
        ByZoningGrouping(
            x_intervals=[0.0, 2.001], name_template="full_b"
        ),
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
    verts, _normals, parents_out = slice_triangles_with_parents(
        tri, normals, parents, intervals
    )
    # The triangle straddles the x=1 plane, so it slices.
    assert verts.shape[0] >= 2
    # Every fragment retains the parent index.
    assert np.all(parents_out == 42)


def test_two_container_connectivity_split(two_container_mesh):
    """Connectivity isolates the two containers as separate groups."""
    chain = [
        ByConnectivityGrouping(name_template="container_{idx}", min_triangles=4)
    ]
    grouping = build_regroup_mapping(
        two_container_mesh, chain, transformation=None
    )
    assert len(grouping.groups) == 2
    sizes = sorted(int(idxs.size) for idxs in grouping.groups.values())
    # Container A: 4*6*2 = 48 triangles; Container B: 2*3*2 = 12 triangles.
    assert sizes == [12, 48]
