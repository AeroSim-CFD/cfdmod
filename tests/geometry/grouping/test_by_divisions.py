"""Tests for ByDivisionsGrouping."""

from __future__ import annotations

import pytest

from cfdmod.geometry import (
    ByDivisionsGrouping,
    BySurfaceGrouping,
    apply_groupings,
)


def test_three_x_divisions_assign_each_triangle_to_a_cell(grid_mesh):
    # grid_mesh: x-centroids at 0.5, 1.5, 2.5; one surface "S".
    # 3 equal x-cells over the centroid bbox [0.5, 2.5] -> two triangles each.
    spec = ByDivisionsGrouping(n_div_x=3)
    res = apply_groupings(grid_mesh, [spec])
    assert set(res.groups) == {"r0", "r1", "r2"}
    assert sorted(res.groups["r0"].tolist()) == [0, 1]
    assert sorted(res.groups["r1"].tolist()) == [2, 3]
    assert sorted(res.groups["r2"].tolist()) == [4, 5]


def test_max_coordinate_centroid_is_included(grid_mesh):
    # The max-x centroid sits exactly on the upper bbox edge; padding the
    # last interval up via nextafter must keep it inside cell r1 (not dropped).
    spec = ByDivisionsGrouping(n_div_x=2)
    res = apply_groupings(grid_mesh, [spec])
    assigned = set()
    for idxs in res.groups.values():
        assigned.update(idxs.tolist())
    assert assigned == {0, 1, 2, 3, 4, 5}


def test_none_axis_means_no_binning_along_that_axis(grid_mesh):
    # n_div_x=None, n_div_y=None, n_div_z=None -> single cell, all triangles.
    spec = ByDivisionsGrouping()
    res = apply_groupings(grid_mesh, [spec])
    assert set(res.groups) == {"r0"}
    assert sorted(res.groups["r0"].tolist()) == [0, 1, 2, 3, 4, 5]


def test_name_template_with_axis_indices(grid_mesh):
    spec = ByDivisionsGrouping(n_div_x=3, name_template="x{ix}")
    res = apply_groupings(grid_mesh, [spec])
    assert set(res.groups) == {"x0", "x1", "x2"}


def test_restrict_to_uses_restricted_bbox(two_square_mesh):
    # Surface A covers x in [0,1] (centroids ~0.5,0.5), B covers x in [2,3].
    # Restrict divisions to A only: bbox is just A, so n_div_x=2 splits the
    # x range of A, not the full mesh.
    surfs = BySurfaceGrouping(sets={"a_only": ["A"]})
    div = ByDivisionsGrouping(n_div_x=2, name_template="A{ix}", restrict_to=["a_only"])
    res = apply_groupings(two_square_mesh, [surfs, div])
    # Only the two triangles of surface A may appear in A0/A1.
    a_tris = set(res.groups["a_only"].tolist())
    seen = set()
    for name in ("A0", "A1"):
        if name in res.groups:
            seen.update(res.groups[name].tolist())
    assert seen == a_tris


def test_name_template_collision_raises(grid_mesh):
    spec = ByDivisionsGrouping(n_div_x=3, name_template="constant")
    with pytest.raises(ValueError, match="duplicate group name"):
        apply_groupings(grid_mesh, [spec])


def test_n_div_must_be_positive():
    with pytest.raises(ValueError):
        ByDivisionsGrouping(n_div_x=0)
