"""Tests for BySurfaceGrouping."""

from __future__ import annotations

import pytest

from cfdmod.geometry import BySurfaceGrouping, apply_groupings


def test_single_set_unions_listed_surfaces(two_square_mesh):
    spec = BySurfaceGrouping(sets={"both": ["A", "B"]})
    res = apply_groupings(two_square_mesh, [spec])
    assert set(res.groups) == {"both"}
    assert sorted(res.groups["both"].tolist()) == [0, 1, 2, 3]


def test_separate_sets_keep_separate_groups(two_square_mesh):
    spec = BySurfaceGrouping(sets={"left": ["A"], "right": ["B"]})
    res = apply_groupings(two_square_mesh, [spec])
    assert set(res.groups) == {"left", "right"}
    assert res.groups["left"].tolist() == [0, 1]
    assert res.groups["right"].tolist() == [2, 3]


def test_include_unlisted_picks_up_remaining_surfaces(two_square_mesh):
    spec = BySurfaceGrouping(sets={"only_a": ["A"]}, include_unlisted=True)
    res = apply_groupings(two_square_mesh, [spec])
    # 'B' becomes a singleton group keyed by its surface name.
    assert set(res.groups) == {"only_a", "B"}
    assert res.groups["B"].tolist() == [2, 3]


def test_unknown_surface_raises(two_square_mesh):
    spec = BySurfaceGrouping(sets={"bad": ["does_not_exist"]})
    with pytest.raises(KeyError, match="surfaces not in mesh"):
        apply_groupings(two_square_mesh, [spec])


def test_duplicate_surface_in_set_rejected_by_validator():
    with pytest.raises(ValueError, match="surface names must be unique"):
        BySurfaceGrouping(sets={"x": ["A", "A"]})
