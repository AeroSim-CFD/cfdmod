"""Unit tests for the filter_by_grouping op."""

from __future__ import annotations

import numpy as np
import pytest

from cfdmod.adapters.memory import MemoryFieldStore
from cfdmod.core import (
    ElementMeta,
    Grouping,
    SurfaceDataSource,
    TimeAxis,
    Topology,
)
from cfdmod.core.ops.data_source_create import FilterByGroupingParams, filter_by_grouping


def _surface_with_grouping() -> SurfaceDataSource:
    verts = np.array(
        [[0, 0, 0], [1, 0, 0], [0, 1, 0], [1, 1, 0], [0.5, 0.5, 1]], dtype=np.float64
    )
    tris = np.array([[0, 1, 2], [1, 3, 2], [2, 3, 4], [0, 2, 4]], dtype=np.int32)
    pressure = np.arange(4 * 3, dtype=np.float64).reshape(4, 3)
    return SurfaceDataSource(
        time=TimeAxis(initial_time=0.0, timestep_size=1.0, n_timesteps=3),
        topology=Topology.triangles(tris, verts),
        elements=ElementMeta(area=np.array([1.0, 2.0, 3.0, 4.0])),
        groupings={"body": Grouping(name="body", indices=[0, 0, 1, 1])},
        fields=MemoryFieldStore({"pressure": pressure}),
    )


def test_filter_keep_keeps_only_listed_groups():
    ds = _surface_with_grouping()
    out = filter_by_grouping(ds, FilterByGroupingParams(grouping="body", keep=[0]))
    assert out.n_elements == 2
    np.testing.assert_array_equal(
        out.fields.read("pressure"), ds.fields.read("pressure")[[0, 1]]
    )
    np.testing.assert_array_equal(out.elements.area, [1.0, 2.0])
    # Surviving groupings are sliced consistently.
    np.testing.assert_array_equal(out.groupings["body"].indices, [0, 0])


def test_filter_drop_removes_listed_groups():
    ds = _surface_with_grouping()
    out = filter_by_grouping(ds, FilterByGroupingParams(grouping="body", drop=[1]))
    assert out.n_elements == 2


def test_filter_requires_exactly_one_of_keep_or_drop():
    ds = _surface_with_grouping()
    with pytest.raises(ValueError):
        filter_by_grouping(ds, FilterByGroupingParams(grouping="body"))
    with pytest.raises(ValueError):
        filter_by_grouping(ds, FilterByGroupingParams(grouping="body", keep=[0], drop=[1]))


def test_filter_unknown_grouping_raises_keyerror():
    ds = _surface_with_grouping()
    with pytest.raises(KeyError):
        filter_by_grouping(ds, FilterByGroupingParams(grouping="missing", keep=[0]))


def test_filter_empty_selection_raises():
    ds = _surface_with_grouping()
    with pytest.raises(ValueError):
        filter_by_grouping(ds, FilterByGroupingParams(grouping="body", keep=[99]))
