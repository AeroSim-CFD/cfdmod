"""Exhaustive edge-case coverage for grouping-spec validators.

Each field-validator branch that raises is exercised here (interval
degeneracies, range checks, empty names), plus the apply-time
duplicate-name guard and the zero-extent size-binning branch. These
were only partially covered by the per-kind test files.
"""

from __future__ import annotations

import pytest

from cfdmod.geometry import (
    ByCylindricalGrouping,
    ByPercentileGrouping,
    BySurfaceGrouping,
    ByZoningGrouping,
    apply_groupings,
)
from cfdmod.geometry.grouping.kinds.by_size import _intervals_from_size

# --- ByZoningGrouping._validate_intervals (x / y / z) ----------------------


def test_zoning_intervals_need_at_least_two_values():
    # A single value is ambiguous; an empty list is the valid "no binning"
    # sentinel, so only the single-value case must raise.
    with pytest.raises(ValueError, match="at least 2 values"):
        ByZoningGrouping(x_intervals=[1.0])


def test_zoning_empty_intervals_are_no_bin_sentinel():
    spec = ByZoningGrouping(x_intervals=[])
    assert spec.x_intervals == [float("-inf"), float("inf")]


def test_zoning_intervals_reject_repeats():
    with pytest.raises(ValueError, match="must not repeat"):
        ByZoningGrouping(y_intervals=[0.0, 0.0, 1.0])


def test_zoning_intervals_must_be_ascending():
    with pytest.raises(ValueError, match="strictly ascending"):
        ByZoningGrouping(z_intervals=[1.0, 0.0])


# --- ByCylindricalGrouping interval validators -----------------------------

_ORIGIN = (0.0, 0.0, 0.0)


def test_cylindrical_r_needs_two_values():
    with pytest.raises(ValueError, match="r_intervals must have at least 2"):
        ByCylindricalGrouping(origin=_ORIGIN, r_intervals=[1.0])


def test_cylindrical_r_rejects_negative():
    with pytest.raises(ValueError, match="r_intervals must be non-negative"):
        ByCylindricalGrouping(origin=_ORIGIN, r_intervals=[-1.0, 1.0])


def test_cylindrical_r_must_ascend():
    with pytest.raises(ValueError, match="r_intervals must be strictly ascending"):
        ByCylindricalGrouping(origin=_ORIGIN, r_intervals=[2.0, 1.0])


def test_cylindrical_theta_needs_two_values():
    with pytest.raises(ValueError, match="theta_intervals_deg must have at least 2"):
        ByCylindricalGrouping(origin=_ORIGIN, theta_intervals_deg=[0.0])


def test_cylindrical_theta_must_ascend():
    with pytest.raises(ValueError, match="theta_intervals_deg must be strictly ascending"):
        ByCylindricalGrouping(origin=_ORIGIN, theta_intervals_deg=[90.0, 10.0])


def test_cylindrical_axial_needs_two_values():
    with pytest.raises(ValueError, match="axial_intervals must have at least 2"):
        ByCylindricalGrouping(origin=_ORIGIN, axial_intervals=[0.0])


def test_cylindrical_axial_must_ascend():
    with pytest.raises(ValueError, match="axial_intervals must be strictly ascending"):
        ByCylindricalGrouping(origin=_ORIGIN, axial_intervals=[1.0, 0.0])


# --- BySurfaceGrouping.sets ------------------------------------------------


def test_surface_empty_group_name_rejected():
    with pytest.raises(ValueError, match="group name must be non-empty"):
        BySurfaceGrouping(sets={"": ["S"]})


# --- ByPercentileGrouping duplicate-name guard (apply time) ----------------


def test_percentile_name_template_without_idx_collides(grid_mesh):
    spec = ByPercentileGrouping(axis="x", n_quantiles=3, name_template="fixed")
    with pytest.raises(ValueError, match="duplicate group name"):
        apply_groupings(grid_mesh, [spec])


# --- BySizeGrouping zero-extent edge (all-equal centroids) -----------------


def test_intervals_from_size_zero_extent_is_single_cell():
    # lo == hi (a degenerate axis, e.g. all centroids share a coordinate)
    # must collapse to exactly one cell, not crash or produce empty edges.
    edges = _intervals_from_size(5.0, 5.0, 1.0)
    assert len(edges) == 2
    assert edges[0] <= 5.0 < edges[1]


def test_intervals_from_size_none_is_no_bin_sentinel():
    edges = _intervals_from_size(0.0, 3.0, None)
    assert edges == [float("-inf"), float("inf")]
