"""Tests for the apply_groupings driver itself (composition, errors)."""

from __future__ import annotations

import pytest

from cfdmod.geometry import (
    ByConnectivityGrouping,
    BySurfaceGrouping,
    ByZoningGrouping,
    apply_groupings,
)


def test_empty_pipeline_raises(two_square_mesh):
    with pytest.raises(ValueError, match="empty"):
        apply_groupings(two_square_mesh, [])


def test_name_collision_between_specs_raises(two_square_mesh):
    spec_a = BySurfaceGrouping(sets={"x": ["A"]})
    spec_b = BySurfaceGrouping(sets={"x": ["B"]})
    with pytest.raises(ValueError, match="name collision"):
        apply_groupings(two_square_mesh, [spec_a, spec_b])


def test_parent_n_triangles_recorded(two_square_mesh):
    spec = BySurfaceGrouping(sets={"all": ["A", "B"]})
    res = apply_groupings(two_square_mesh, [spec])
    assert res.parent_n_triangles == 4


def test_by_connectivity_step1_raises_not_implemented(two_square_mesh):
    spec = ByConnectivityGrouping()
    with pytest.raises(NotImplementedError, match="step 2"):
        apply_groupings(two_square_mesh, [spec])
