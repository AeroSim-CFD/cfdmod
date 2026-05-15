"""Tests for ByPercentileGrouping."""

from __future__ import annotations

import pytest

from cfdmod.geometry import (
    ByPercentileGrouping,
    BySurfaceGrouping,
    apply_groupings,
)


def test_three_quantiles_balance_two_per_bucket(grid_mesh):
    # Centroid x's: 0.667, 0.333, 1.667, 1.333, 2.667, 2.333. With 3
    # equal-count quantiles, each bucket holds exactly two centroids,
    # paired by adjacency in the sorted distribution.
    spec = ByPercentileGrouping(axis="x", n_quantiles=3)
    res = apply_groupings(grid_mesh, [spec])
    assert set(res.groups) == {"q0", "q1", "q2"}
    assert sorted(res.groups["q0"].tolist()) == [0, 1]
    assert sorted(res.groups["q1"].tolist()) == [2, 3]
    assert sorted(res.groups["q2"].tolist()) == [4, 5]


def test_n_quantiles_one_returns_single_bucket(grid_mesh):
    spec = ByPercentileGrouping(axis="x", n_quantiles=1)
    res = apply_groupings(grid_mesh, [spec])
    assert set(res.groups) == {"q0"}
    assert sorted(res.groups["q0"].tolist()) == [0, 1, 2, 3, 4, 5]


def test_max_coordinate_centroid_is_included(grid_mesh):
    # Quantile binning pads the upper edge with nextafter so the max
    # centroid is not dropped at the boundary.
    spec = ByPercentileGrouping(axis="x", n_quantiles=2)
    res = apply_groupings(grid_mesh, [spec])
    assigned = set()
    for idxs in res.groups.values():
        assigned.update(idxs.tolist())
    assert assigned == {0, 1, 2, 3, 4, 5}


def test_restrict_to_uses_restricted_distribution(two_square_mesh):
    surfs = BySurfaceGrouping(sets={"a_only": ["A"]})
    spec = ByPercentileGrouping(
        axis="x",
        n_quantiles=2,
        restrict_to=["a_only"],
        name_template="A{idx}",
    )
    res = apply_groupings(two_square_mesh, [surfs, spec])
    a_tris = set(res.groups["a_only"].tolist())
    seen = set()
    for name in res.groups:
        if name.startswith("A"):
            seen.update(res.groups[name].tolist())
    assert seen == a_tris


def test_n_quantiles_must_be_positive():
    with pytest.raises(ValueError):
        ByPercentileGrouping(axis="x", n_quantiles=0)


def test_axis_must_be_one_of_x_y_z():
    with pytest.raises(ValueError):
        ByPercentileGrouping(axis="w", n_quantiles=3)
