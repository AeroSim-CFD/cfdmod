"""Unit tests for :class:`Grouping` and helpers."""

from __future__ import annotations

import numpy as np

from cfdmod.core import Grouping, elements_in_group, groups_in


def test_groups_in_excludes_ungrouped_by_default():
    g = Grouping(name="surf", indices=[0, 0, 1, -1, 2, 2])
    assert np.array_equal(groups_in(g), [0, 1, 2])
    assert np.array_equal(groups_in(g, include_ungrouped=True), [-1, 0, 1, 2])


def test_elements_in_group_returns_matching_indices():
    g = Grouping(name="surf", indices=[0, 0, 1, -1, 2, 2])
    assert np.array_equal(elements_in_group(g, 0), [0, 1])
    assert np.array_equal(elements_in_group(g, 2), [4, 5])
    assert elements_in_group(g, 7).size == 0


def test_label_resolves_via_id_to_label_or_falls_back():
    g = Grouping(
        name="surf",
        indices=[0, 1, 2],
        id_to_label={0: "front", 2: "back"},
    )
    assert g.label(0) == "front"
    assert g.label(2) == "back"
    assert g.label(1) == "1"
