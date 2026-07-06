"""Unit tests for the regroup-pipeline extensions to the grouping pipeline."""

from __future__ import annotations

import numpy as np
import pytest

from cfdmod.geometry.grouping import (
    BySizeRoundedPerComponent,
    apply_groupings,
    expand_size_rounded_chain,
)
from cfdmod.geometry.grouping.kinds.by_divisions import ByDivisionsGrouping
from cfdmod.geometry.grouping.kinds.by_surface import BySurfaceGrouping
from cfdmod.geometry.grouping.regroup import _round_half_up


def test_round_half_up_rounds_half_away_from_zero():
    assert _round_half_up(0.5) == 1
    assert _round_half_up(1.5) == 2
    assert _round_half_up(2.5) == 3  # banker's rounding would give 2
    assert _round_half_up(0.49) == 0
    assert _round_half_up(0.51) == 1


def test_expand_passes_canonical_specs_through_unchanged(grid_mesh):
    chain = [BySurfaceGrouping(sets={"all": ["S"]})]
    expanded = expand_size_rounded_chain(grid_mesh, chain)
    assert expanded == chain


def test_expand_size_rounded_appends_one_division_spec_per_parent(two_square_mesh):
    chain = [
        BySurfaceGrouping(sets={"left": ["A"], "right": ["B"]}),
        BySizeRoundedPerComponent(target_size_x=0.5, name_template="{parent}_r{idx}"),
    ]
    expanded = expand_size_rounded_chain(two_square_mesh, chain)

    assert len(expanded) == 3
    assert isinstance(expanded[0], BySurfaceGrouping)

    div_specs = expanded[1:]
    for spec in div_specs:
        assert isinstance(spec, ByDivisionsGrouping)
        assert spec.n_div_x == 1  # extent ~ 0 for centroids of one row of squares
        assert spec.n_div_y is None
        assert spec.n_div_z is None
    assert {tuple(s.restrict_to) for s in div_specs} == {("left",), ("right",)}


def test_expand_uses_per_parent_bbox_for_division_count(two_square_mesh):
    """Wider parents get more cells than narrower parents at the same target size."""
    bigger_squares = two_square_mesh.copy()
    # Stretch surface B to span x in [2, 6] (extent 4) by translating its
    # second triangle outward.
    geom = bigger_squares.geometry
    verts = geom.vertices.copy()
    b_tri_idx = bigger_squares.surfaces["B"]
    b_vert_ids = np.unique(geom.triangles[b_tri_idx].reshape(-1))
    # Shift the rightmost x of surface B from 3 to 8 -> centroid extent of 2.0.
    for vid in b_vert_ids:
        if verts[vid, 0] == 3.0:
            verts[vid, 0] = 8.0
    bigger_squares.geometry = type(geom)(vertices=verts, triangles=geom.triangles)

    chain = [
        BySurfaceGrouping(sets={"left": ["A"], "right": ["B"]}),
        BySizeRoundedPerComponent(target_size_x=1.0, name_template="{parent}_r{idx}"),
    ]
    expanded = expand_size_rounded_chain(bigger_squares, chain)
    div_by_parent = {tuple(s.restrict_to): s.n_div_x for s in expanded[1:]}
    # left centroids: original [0,1] square, two triangles, centroids at
    # x=2/3 and x=1/3 -> extent 1/3 -> round(1/3) = 0 -> floored to 1.
    # right centroids: shifted [2,8] square, centroids at x=6 and x=4 ->
    # extent 2.0 -> round(2/1.0) = 2.
    assert div_by_parent[("left",)] == 1
    assert div_by_parent[("right",)] == 2


def test_expand_raises_when_size_rounded_is_first(grid_mesh):
    chain = [BySizeRoundedPerComponent(target_size_x=0.5)]
    with pytest.raises(ValueError, match="no prior chain"):
        expand_size_rounded_chain(grid_mesh, chain)


def test_min_n_div_floors_the_count(grid_mesh):
    chain = [
        BySurfaceGrouping(sets={"all": ["S"]}),
        BySizeRoundedPerComponent(target_size_x=100.0, min_n_div=3),
    ]
    expanded = expand_size_rounded_chain(grid_mesh, chain)
    div = expanded[1]
    assert isinstance(div, ByDivisionsGrouping)
    assert div.n_div_x == 3


def test_expanded_chain_is_runnable_through_apply_groupings(two_square_mesh):
    chain = [
        BySurfaceGrouping(sets={"left": ["A"], "right": ["B"]}),
        BySizeRoundedPerComponent(target_size_x=10.0, name_template="{parent}_c{idx}"),
    ]
    expanded = expand_size_rounded_chain(two_square_mesh, chain)
    result = apply_groupings(two_square_mesh, expanded)

    assert "left" in result.groups
    assert "right" in result.groups
    sub_group_names = [n for n in result.groups if n.endswith("_c0")]
    assert {"left_c0", "right_c0"} == set(sub_group_names)
    np.testing.assert_array_equal(
        np.sort(result.groups["left_c0"]),
        np.sort(result.groups["left"]),
    )
