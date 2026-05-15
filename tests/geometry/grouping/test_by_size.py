"""Tests for BySizeGrouping."""

from __future__ import annotations

import pytest

from cfdmod.geometry import (
    BySizeGrouping,
    BySurfaceGrouping,
    apply_groupings,
)


def test_size_smaller_than_centroid_spacing_makes_one_cell_per_centroid(grid_mesh):
    # grid_mesh: x-centroids at 0.5, 1.5, 2.5; extent=2.0. size=0.8 ->
    # ceil(2.0/0.8)=3 cells anchored at 0.5: edges 0.5, 1.3, 2.1, 2.9.
    # Each centroid lands in its own cell.
    spec = BySizeGrouping(size_x=0.8)
    res = apply_groupings(grid_mesh, [spec])
    assert set(res.groups) == {"r0", "r1", "r2"}
    assert sorted(res.groups["r0"].tolist()) == [0, 1]
    assert sorted(res.groups["r1"].tolist()) == [2, 3]
    assert sorted(res.groups["r2"].tolist()) == [4, 5]


def test_size_equal_to_unit_square_yields_one_cell_per_square(grid_mesh):
    # Centroid x-bbox is roughly [0.333, 2.667] (extent ~2.333).
    # size_x=1.0 -> ceil(2.333)=3 cells of width 1.0 anchored at bbox_min;
    # each unit square's two triangles end up in the same cell.
    spec = BySizeGrouping(size_x=1.0)
    res = apply_groupings(grid_mesh, [spec])
    assert set(res.groups) == {"r0", "r1", "r2"}
    assert sorted(res.groups["r0"].tolist()) == [0, 1]
    assert sorted(res.groups["r1"].tolist()) == [2, 3]
    assert sorted(res.groups["r2"].tolist()) == [4, 5]


def test_size_larger_than_extent_collapses_to_one_cell(grid_mesh):
    # x-extent of centroids is 2.0; size 10.0 -> single cell with everything.
    spec = BySizeGrouping(size_x=10.0)
    res = apply_groupings(grid_mesh, [spec])
    assert set(res.groups) == {"r0"}
    assert sorted(res.groups["r0"].tolist()) == [0, 1, 2, 3, 4, 5]


def test_non_dividing_size_includes_max_centroid(grid_mesh):
    # Extent 2.0, size 0.7 -> ceil(2.0/0.7) = 3 cells; last cell upper edge
    # padded so the max-x centroid (at 2.5) is included.
    spec = BySizeGrouping(size_x=0.7)
    res = apply_groupings(grid_mesh, [spec])
    assigned = set()
    for idxs in res.groups.values():
        assigned.update(idxs.tolist())
    assert assigned == {0, 1, 2, 3, 4, 5}


def test_none_axis_means_no_binning_along_that_axis(grid_mesh):
    spec = BySizeGrouping()
    res = apply_groupings(grid_mesh, [spec])
    assert set(res.groups) == {"r0"}
    assert sorted(res.groups["r0"].tolist()) == [0, 1, 2, 3, 4, 5]


def test_restrict_to_uses_restricted_bbox(two_square_mesh):
    surfs = BySurfaceGrouping(sets={"a_only": ["A"]})
    sized = BySizeGrouping(size_x=0.5, name_template="A{ix}", restrict_to=["a_only"])
    res = apply_groupings(two_square_mesh, [surfs, sized])
    a_tris = set(res.groups["a_only"].tolist())
    seen = set()
    for name in res.groups:
        if name.startswith("A"):
            seen.update(res.groups[name].tolist())
    assert seen == a_tris


def test_size_must_be_positive():
    with pytest.raises(ValueError):
        BySizeGrouping(size_x=0.0)
    with pytest.raises(ValueError):
        BySizeGrouping(size_x=-1.0)
