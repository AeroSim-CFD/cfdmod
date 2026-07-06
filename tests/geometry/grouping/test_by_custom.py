"""Tests for CustomGrouping."""

from __future__ import annotations

import numpy as np
import pytest

from cfdmod.geometry import (
    BySurfaceGrouping,
    CustomGrouping,
    apply_groupings,
    dump_groupings,
    load_groupings,
)

from tests.geometry.grouping._custom_callbacks import first_n, split_by_threshold

_DOTTED = "tests.geometry.grouping._custom_callbacks.split_by_threshold"


def test_inline_callable_runs_and_returns_groups(grid_mesh):
    # Centroids x: 0.667, 0.333, 1.667, 1.333, 2.667, 2.333
    spec = CustomGrouping(
        callback=split_by_threshold,
        params={"axis": "x", "threshold": 1.5},
    )
    res = apply_groupings(grid_mesh, [spec])
    assert sorted(res.groups["below"].tolist()) == [0, 1, 3]
    assert sorted(res.groups["above"].tolist()) == [2, 4, 5]


def test_dotted_path_callback_resolves(grid_mesh):
    spec = CustomGrouping(
        callback=_DOTTED,
        params={"axis": "x", "threshold": 1.5},
    )
    res = apply_groupings(grid_mesh, [spec])
    assert sorted(res.groups["below"].tolist()) == [0, 1, 3]
    assert sorted(res.groups["above"].tolist()) == [2, 4, 5]


def test_params_passed_through(grid_mesh):
    spec = CustomGrouping(callback=first_n, params={"n": 4, "name": "head"})
    res = apply_groupings(grid_mesh, [spec])
    assert sorted(res.groups["head"].tolist()) == [0, 1, 2, 3]


def test_restrict_to_filters_candidates(two_square_mesh):
    surfs = BySurfaceGrouping(sets={"a_only": ["A"]})
    spec = CustomGrouping(
        callback=first_n,
        params={"n": 100, "name": "all_in_a"},
        restrict_to=["a_only"],
    )
    res = apply_groupings(two_square_mesh, [surfs, spec])
    assert sorted(res.groups["all_in_a"].tolist()) == sorted(res.groups["a_only"].tolist())


def test_callback_returning_out_of_range_raises(grid_mesh):
    def bad(mesh, cand, params):
        return {"x": np.array([99], dtype=np.int64)}

    with pytest.raises(ValueError, match="out of range"):
        apply_groupings(grid_mesh, [CustomGrouping(callback=bad)])


def test_callback_returning_outside_restricted_set_raises(two_square_mesh):
    def bad(mesh, cand, params):
        # Returns triangle 2 (in surface B) even though restrict_to=A.
        return {"x": np.array([0, 1, 2], dtype=np.int64)}

    surfs = BySurfaceGrouping(sets={"a_only": ["A"]})
    spec = CustomGrouping(callback=bad, restrict_to=["a_only"])
    with pytest.raises(ValueError, match="restrict_to violation"):
        apply_groupings(two_square_mesh, [surfs, spec])


def test_non_dict_return_raises(grid_mesh):
    def bad(mesh, cand, params):
        return [0, 1, 2]

    with pytest.raises(TypeError, match="dict"):
        apply_groupings(grid_mesh, [CustomGrouping(callback=bad)])


def test_non_string_group_name_raises(grid_mesh):
    def bad(mesh, cand, params):
        return {42: np.array([0, 1])}

    with pytest.raises(TypeError, match="str"):
        apply_groupings(grid_mesh, [CustomGrouping(callback=bad)])


def test_dotted_path_round_trip():
    chain = [CustomGrouping(callback=_DOTTED, params={"axis": "x", "threshold": 1.5})]
    serialised = dump_groupings(chain)
    assert serialised[0]["callback"] == _DOTTED
    rehydrated = load_groupings(serialised)
    assert rehydrated[0].callback == _DOTTED
    assert rehydrated[0].params == {"axis": "x", "threshold": 1.5}


def test_callable_serialised_to_dotted_path_when_importable():
    chain = [CustomGrouping(callback=split_by_threshold, params={"axis": "x"})]
    serialised = dump_groupings(chain)
    assert serialised[0]["callback"] == _DOTTED
    rehydrated = load_groupings(serialised)
    assert rehydrated[0].callback == _DOTTED


def test_lambda_cannot_be_serialised():
    spec = CustomGrouping(callback=lambda mesh, cand, params: {"x": cand})
    with pytest.raises(ValueError, match="not importable"):
        dump_groupings([spec])


def test_validation_rejects_non_callable_non_string():
    with pytest.raises(ValueError):
        CustomGrouping(callback=42)


def test_validation_rejects_string_without_dot():
    with pytest.raises(ValueError):
        CustomGrouping(callback="just_a_word")


def test_unresolvable_dotted_path_raises_at_apply(grid_mesh):
    spec = CustomGrouping(callback="cfdmod.does_not_exist_anywhere")
    with pytest.raises((ValueError, ModuleNotFoundError, AttributeError)):
        apply_groupings(grid_mesh, [spec])
