"""Tests for ByPlaneGrouping."""

from __future__ import annotations

import pytest

from cfdmod.geometry import ByPlaneGrouping, BySurfaceGrouping, apply_groupings


def test_default_intervals_split_into_two_half_spaces(grid_mesh):
    # Plane at x=1.5, normal +x. Centroids x ~ {0.333, 0.667, 1.333, 1.667,
    # 2.333, 2.667}; first three are negative side, last three positive.
    spec = ByPlaneGrouping(point=(1.5, 0.0, 0.0), normal=(1.0, 0.0, 0.0))
    res = apply_groupings(grid_mesh, [spec])
    assert set(res.groups) == {"r0", "r1"}
    # r0 is the [-inf, 0) half-space (signed distance < 0): tris 0, 1, 3
    # (centroids 0.667, 0.333, 1.333). Tri 2 has centroid 1.667 > 1.5 so
    # is in r1.
    assert sorted(res.groups["r0"].tolist()) == [0, 1, 3]
    assert sorted(res.groups["r1"].tolist()) == [2, 4, 5]


def test_oblique_plane_partitions_correctly(grid_mesh):
    # Plane normal (1, 1, 0) (auto-normalised); point at origin.
    # Signed distance for centroid (cx, cy, 0) is (cx + cy) / sqrt(2),
    # always positive for grid_mesh (cx,cy > 0). All triangles in r1.
    spec = ByPlaneGrouping(point=(0.0, 0.0, 0.0), normal=(1.0, 1.0, 0.0))
    res = apply_groupings(grid_mesh, [spec])
    assert set(res.groups) == {"r1"}
    assert sorted(res.groups["r1"].tolist()) == [0, 1, 2, 3, 4, 5]


def test_explicit_intervals_carve_three_slabs(grid_mesh):
    spec = ByPlaneGrouping(
        point=(0.0, 0.0, 0.0),
        normal=(1.0, 0.0, 0.0),
        intervals=[0.0, 1.0, 2.0, 3.0],
    )
    res = apply_groupings(grid_mesh, [spec])
    # Slab 0: x in [0,1) -> tris 0, 1; slab 1: [1,2) -> 2, 3; slab 2: [2,3) -> 4, 5.
    assert sorted(res.groups["r0"].tolist()) == [0, 1]
    assert sorted(res.groups["r1"].tolist()) == [2, 3]
    assert sorted(res.groups["r2"].tolist()) == [4, 5]


def test_restrict_to_scopes_the_plane_split(two_square_mesh):
    surfs = BySurfaceGrouping(sets={"a_only": ["A"]})
    plane = ByPlaneGrouping(
        point=(0.5, 0.0, 0.0),
        normal=(1.0, 0.0, 0.0),
        restrict_to=["a_only"],
        name_template="A{idx}",
    )
    res = apply_groupings(two_square_mesh, [surfs, plane])
    a_tris = set(res.groups["a_only"].tolist())
    seen = set()
    for name in res.groups:
        if name.startswith("A"):
            seen.update(res.groups[name].tolist())
    assert seen == a_tris


def test_zero_normal_is_rejected():
    with pytest.raises(ValueError):
        ByPlaneGrouping(point=(0, 0, 0), normal=(0, 0, 0))


def test_intervals_must_be_strictly_ascending():
    with pytest.raises(ValueError):
        ByPlaneGrouping(point=(0, 0, 0), normal=(1, 0, 0), intervals=[1.0, 0.0])
    with pytest.raises(ValueError):
        ByPlaneGrouping(point=(0, 0, 0), normal=(1, 0, 0), intervals=[0.0, 0.0])
