"""Chunked time executor: parity with whole-series and correct windowing."""

from __future__ import annotations

import numpy as np
import pytest

from cfdmod.adapters.memory import MemoryFieldStore
from cfdmod.core.chunked import (
    assert_time_chunkable,
    chunk_map_time,
    concat_time,
    slice_time,
    time_windows,
)
from cfdmod.core.data_source import SurfaceDataSource
from cfdmod.core.field_meta import FieldMeta
from cfdmod.core.grouping import Grouping
from cfdmod.core.ops.data_source_create.field_series_for_groups import (
    FieldSeriesForGroupsParams,
    field_series_for_groups,
)
from cfdmod.core.ops.field.algebra import ScaleParams, scale
from cfdmod.core.time_axis import TimeAxis
from cfdmod.core.topology import ElementMeta, Topology


def _surface(n_elements: int, n_t: int, seed: int = 0) -> SurfaceDataSource:
    rng = np.random.default_rng(seed)
    verts = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0], [1, 1, 0]], dtype=np.float64)
    tris = np.array([[0, 1, 2], [1, 3, 2]], dtype=np.int32)
    tris = np.tile(tris, (max(n_elements // 2, 1), 1))[:n_elements]
    field = rng.standard_normal((n_elements, n_t))
    return SurfaceDataSource(
        time=TimeAxis(initial_time=1.0, timestep_size=0.5, n_timesteps=n_t),
        topology=Topology.triangles(tris, verts),
        elements=ElementMeta(),
        fields=MemoryFieldStore({"cp": field}),
        field_meta={"cp": FieldMeta(name="cp")},
    )


@pytest.mark.parametrize("chunk_size", [1, 3, 4, 7, 10, None])
def test_scale_pipeline_parity(chunk_size):
    ds = _surface(6, 10)
    pipe = lambda d: scale(d, ScaleParams(field="cp", factor=2.5))  # noqa: E731
    whole = pipe(ds)
    chunked = chunk_map_time(ds, pipe, chunk_size=chunk_size)
    np.testing.assert_allclose(chunked.fields.read("cp"), whole.fields.read("cp"))
    assert chunked.time.n_timesteps == ds.time.n_timesteps


@pytest.mark.parametrize("chunk_size", [1, 2, 3, 5, None])
def test_group_reduction_parity(chunk_size):
    """A reducing pipeline (per-group sum) must match whole-series exactly."""
    ds = _surface(6, 9, seed=3)
    grouping = Grouping(
        name="g",
        indices=np.array([0, 0, 1, 1, 2, 2], dtype=np.int32),
        id_to_label={0: "a", 1: "b", 2: "c"},
    )
    ds = ds.with_grouping(grouping)
    pipe = lambda d: field_series_for_groups(  # noqa: E731
        d, FieldSeriesForGroupsParams(grouping="g", field="cp", agg="sum", out="cp")
    )
    whole = pipe(ds)
    chunked = chunk_map_time(ds, pipe, chunk_size=chunk_size)
    assert chunked.kind == "groups"
    assert chunked.n_elements == 3
    np.testing.assert_allclose(chunked.fields.read("cp"), whole.fields.read("cp"))


def test_time_windows_cover_all():
    assert list(time_windows(10, 4)) == [slice(0, 4), slice(4, 8), slice(8, 10)]
    assert list(time_windows(8, 4)) == [slice(0, 4), slice(4, 8)]
    with pytest.raises(ValueError):
        list(time_windows(10, 0))


def test_slice_time_preserves_axis_and_offset():
    ds = _surface(4, 10)
    win = slice_time(ds, slice(4, 7))
    assert win.time.n_timesteps == 3
    np.testing.assert_allclose(win.time.times(), ds.time.times()[4:7])
    np.testing.assert_allclose(win.fields.read("cp"), ds.fields.read("cp")[:, 4:7])


def test_concat_time_single_and_roundtrip():
    ds = _surface(4, 10)
    parts = [slice_time(ds, sl) for sl in time_windows(10, 4)]
    joined = concat_time(parts)
    np.testing.assert_allclose(joined.fields.read("cp"), ds.fields.read("cp"))
    np.testing.assert_allclose(joined.time.times(), ds.time.times())
    # single part returns as-is
    assert concat_time(parts[:1]) is parts[0]


def test_assert_time_chunkable():
    from cfdmod.core.ops.data_source_create.statistics import StatisticsParams

    # time-chunkable ops pass
    assert_time_chunkable(
        [ScaleParams(field="cp", factor=1.0), FieldSeriesForGroupsParams(grouping="g")]
    )
    # statistics collapses time -> chunkable only along elements, must be rejected
    with pytest.raises(ValueError):
        assert_time_chunkable([StatisticsParams(kinds=["mean"])])
