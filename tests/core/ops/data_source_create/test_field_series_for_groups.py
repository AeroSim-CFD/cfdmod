"""Unit tests for the field_series_for_groups op."""

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
from cfdmod.core.ops.data_source_create import (
    FieldSeriesForGroupsParams,
    field_series_for_groups,
)


def _surface() -> SurfaceDataSource:
    verts = np.array(
        [[0, 0, 0], [1, 0, 0], [0, 1, 0], [1, 1, 0], [0.5, 0.5, 1]], dtype=np.float64
    )
    tris = np.array([[0, 1, 2], [1, 3, 2], [2, 3, 4], [0, 2, 4]], dtype=np.int32)
    pressure = np.array(
        [
            [1.0, 2.0],
            [3.0, 4.0],
            [10.0, 20.0],
            [30.0, 40.0],
        ]
    )
    return SurfaceDataSource(
        time=TimeAxis(initial_time=0.0, timestep_size=1.0, n_timesteps=2),
        topology=Topology.triangles(tris, verts),
        elements=ElementMeta(area=np.array([1.0, 1.0, 2.0, 2.0])),
        groupings={"body": Grouping(name="body", indices=[0, 0, 1, 1])},
        fields=MemoryFieldStore({"pressure": pressure}),
    )


def test_mean_aggregates_per_group_per_timestep():
    ds = _surface()
    out = field_series_for_groups(
        ds, FieldSeriesForGroupsParams(grouping="body", agg="mean")
    )
    assert out.kind == "groups"
    assert out.n_elements == 2  # two groups
    np.testing.assert_allclose(out.fields.read("pressure")[0], [2.0, 3.0])  # mean of rows 0,1
    np.testing.assert_allclose(out.fields.read("pressure")[1], [20.0, 30.0])


def test_sum_aggregation():
    ds = _surface()
    out = field_series_for_groups(
        ds, FieldSeriesForGroupsParams(grouping="body", agg="sum")
    )
    np.testing.assert_allclose(out.fields.read("pressure")[0], [4.0, 6.0])
    np.testing.assert_allclose(out.fields.read("pressure")[1], [40.0, 60.0])


def test_area_weighted_mean_uses_elements_area():
    ds = _surface()
    out = field_series_for_groups(
        ds, FieldSeriesForGroupsParams(grouping="body", agg="area_weighted_mean")
    )
    # Group 0: areas [1, 1] -> simple mean.
    np.testing.assert_allclose(out.fields.read("pressure")[0], [2.0, 3.0])
    # Group 1: areas [2, 2] -> simple mean of [10, 30] / [20, 40].
    np.testing.assert_allclose(out.fields.read("pressure")[1], [20.0, 30.0])


def test_area_weighted_mean_without_area_raises():
    ds = _surface()
    ds = ds.model_copy(update={"elements": ElementMeta()})
    with pytest.raises(ValueError):
        field_series_for_groups(
            ds, FieldSeriesForGroupsParams(grouping="body", agg="area_weighted_mean")
        )


def test_groups_data_source_carries_parent_topology_chain():
    ds = _surface()
    out = field_series_for_groups(
        ds, FieldSeriesForGroupsParams(grouping="body", agg="mean")
    )
    assert out.parent_topology is ds.topology
    assert out.parent_grouping.name == "body"
