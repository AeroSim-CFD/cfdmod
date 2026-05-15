"""Tests for GroupingResult helpers (membership_long, to_region_idx)."""

from __future__ import annotations

import numpy as np

from cfdmod.geometry import GroupingResult


def test_empty_result_membership_long_is_empty():
    res = GroupingResult(parent_n_triangles=10, groups={})
    df = res.membership_long()
    assert df.empty
    assert list(df.columns) == ["triangle_idx", "group_name"]


def test_empty_result_region_idx_is_all_unassigned():
    res = GroupingResult(parent_n_triangles=4, groups={})
    region_idx = res.to_region_idx(unassigned="-")
    assert list(region_idx) == ["-", "-", "-", "-"]


def test_membership_long_one_row_per_pair():
    res = GroupingResult(
        parent_n_triangles=5,
        groups={
            "a": np.array([0, 1, 2], dtype=np.int64),
            "b": np.array([2, 3], dtype=np.int64),
        },
    )
    df = res.membership_long().sort_values(["triangle_idx", "group_name"]).reset_index(drop=True)
    assert df.shape == (5, 2)
    assert df.loc[df["triangle_idx"] == 2, "group_name"].tolist() == ["a", "b"]
    assert df.loc[df["triangle_idx"] == 4, "group_name"].tolist() == []


def test_to_region_idx_joins_overlapping_groups():
    res = GroupingResult(
        parent_n_triangles=4,
        groups={
            "a": np.array([0, 1], dtype=np.int64),
            "b": np.array([1, 2], dtype=np.int64),
        },
    )
    out = res.to_region_idx(sep="|", unassigned="")
    # tri 0 only in 'a'; tri 1 in both; tri 2 only in 'b'; tri 3 in neither.
    assert list(out) == ["a", "a|b", "b", ""]
