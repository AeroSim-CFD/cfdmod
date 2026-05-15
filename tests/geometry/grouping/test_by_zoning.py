"""Tests for ByZoningGrouping."""

from __future__ import annotations

import pytest

from cfdmod.geometry import BySurfaceGrouping, ByZoningGrouping, apply_groupings


def test_single_axis_binning_assigns_each_triangle_to_a_cell(grid_mesh):
    # Three x-cells: [0,1), [1,2), [2,3). Two triangles per square.
    spec = ByZoningGrouping(x_intervals=[0.0, 1.0, 2.0, 3.0])
    res = apply_groupings(grid_mesh, [spec])
    assert set(res.groups) == {"r0", "r1", "r2"}
    assert sorted(res.groups["r0"].tolist()) == [0, 1]
    assert sorted(res.groups["r1"].tolist()) == [2, 3]
    assert sorted(res.groups["r2"].tolist()) == [4, 5]


def test_unbinned_triangles_simply_omitted(grid_mesh):
    # x=[0,2) covers only the first two squares; the third (x in [2,3))
    # has no cell and so contributes to no group.
    spec = ByZoningGrouping(x_intervals=[0.0, 1.0, 2.0])
    res = apply_groupings(grid_mesh, [spec])
    assigned = set()
    for idxs in res.groups.values():
        assigned.update(idxs.tolist())
    assert assigned == {0, 1, 2, 3}


def test_name_template_with_axis_indices(grid_mesh):
    spec = ByZoningGrouping(
        x_intervals=[0.0, 1.0, 2.0, 3.0], name_template="x{ix}"
    )
    res = apply_groupings(grid_mesh, [spec])
    assert set(res.groups) == {"x0", "x1", "x2"}


def test_name_template_collision_raises(grid_mesh):
    # name_template references only ix, but we have three x-cells AND a
    # single y-cell -> still unique. Force collision by ignoring all axes:
    spec = ByZoningGrouping(
        x_intervals=[0.0, 1.0, 2.0, 3.0],
        name_template="constant",
    )
    with pytest.raises(ValueError, match="duplicate group name"):
        apply_groupings(grid_mesh, [spec])


def test_restrict_to_only_bins_named_groups(grid_mesh):
    # First, define two surface-derived groups via a manual zoning that
    # carves the mesh into halves; then bin only the left half.
    half_left = ByZoningGrouping(x_intervals=[0.0, 2.0], name_template="left")
    half_right = ByZoningGrouping(x_intervals=[2.0, 3.0], name_template="right")
    inside_left = ByZoningGrouping(
        x_intervals=[0.0, 1.0, 2.0],
        name_template="L{idx}",
        restrict_to=["left"],
    )
    res = apply_groupings(grid_mesh, [half_left, half_right, inside_left])
    assert set(res.groups) >= {"left", "right", "L0", "L1"}
    # The right-half triangles must NOT appear in the L* groups.
    right_tris = set(res.groups["right"].tolist())
    for name in ("L0", "L1"):
        assert not (set(res.groups[name].tolist()) & right_tris)


def test_restrict_to_unknown_group_raises(grid_mesh):
    spec = ByZoningGrouping(
        x_intervals=[0.0, 1.0, 2.0, 3.0],
        restrict_to=["does_not_exist"],
    )
    with pytest.raises(ValueError, match="unknown groups"):
        apply_groupings(grid_mesh, [spec])


def test_zoning_after_surface_reproduces_legacy_naming(two_square_mesh):
    # Reproduces the legacy "{idx}-{surface_id}" region label pattern by
    # combining BySurfaceGrouping + per-body ByZoningGrouping with
    # restrict_to and a back-compat name_template.
    surfs = BySurfaceGrouping(sets={"body": ["A", "B"]})
    zone = ByZoningGrouping(
        x_intervals=[0.0, 1.0, 2.0, 3.0],
        name_template="{idx}-body",
        restrict_to=["body"],
    )
    res = apply_groupings(two_square_mesh, [surfs, zone])
    # Linear cells: idx 0 = x[0,1), 1 = x[1,2), 2 = x[2,3).
    # Tri 0,1 in 0-body; tri 2,3 in 2-body; nothing in 1-body.
    assert sorted(res.groups["0-body"].tolist()) == [0, 1]
    assert sorted(res.groups["2-body"].tolist()) == [2, 3]
    assert "1-body" not in res.groups
