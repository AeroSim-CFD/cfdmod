"""Unit tests for the statistics op."""

from __future__ import annotations

import numpy as np
import pytest

from cfdmod.adapters.memory import MemoryFieldStore
from cfdmod.core import ElementMeta, SurfaceDataSource, TimeAxis, Topology
from cfdmod.core.ops.data_source_create import StatisticsParams, compute_statistics


def _surface_with_data(data: np.ndarray) -> SurfaceDataSource:
    n_elements, n_t = data.shape
    verts = np.tile(np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0]]), (n_elements, 1)).astype(
        np.float64
    )
    tris = np.arange(n_elements * 3).reshape(n_elements, 3).astype(np.int32)
    return SurfaceDataSource(
        time=TimeAxis(initial_time=0.0, timestep_size=1.0, n_timesteps=n_t),
        topology=Topology.triangles(tris, verts),
        elements=ElementMeta(),
        fields=MemoryFieldStore({"pressure": data.astype(np.float64)}),
    )


def test_statistics_collapses_time_axis():
    rng = np.random.default_rng(0)
    data = rng.normal(size=(4, 100))
    ds = _surface_with_data(data)
    out = compute_statistics(ds, StatisticsParams(kinds=["mean", "rms", "min", "max"]))
    assert out.time.is_time_aggregated
    np.testing.assert_allclose(out.fields.read("mean"), data.mean(axis=1))
    np.testing.assert_allclose(out.fields.read("rms"), data.std(axis=1, ddof=1))
    np.testing.assert_array_equal(out.fields.read("min"), data.min(axis=1))
    np.testing.assert_array_equal(out.fields.read("max"), data.max(axis=1))


def test_statistics_peak_aliases():
    data = np.array([[-1.0, 2.0, -3.0, 4.0]])
    ds = _surface_with_data(data)
    out = compute_statistics(ds, StatisticsParams(kinds=["peak_min", "peak_max"]))
    assert out.fields.read("peak_min")[0] == -3.0
    assert out.fields.read("peak_max")[0] == 4.0


def test_statistics_skewness_kurtosis_match_numpy_definitions():
    rng = np.random.default_rng(1)
    data = rng.gamma(shape=2.0, size=(3, 5000))
    ds = _surface_with_data(data)
    out = compute_statistics(ds, StatisticsParams(kinds=["skewness", "kurtosis"]))
    # Gamma(2) skewness = 2 / sqrt(2) ~= 1.414; excess kurtosis = 6 / 2 = 3.
    assert np.allclose(out.fields.read("skewness").mean(), 2 / np.sqrt(2), atol=0.1)
    assert np.allclose(out.fields.read("kurtosis").mean(), 3.0, atol=0.3)


def test_statistics_rejects_time_aggregated_input():
    verts = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0]], dtype=np.float64)
    tris = np.array([[0, 1, 2]], dtype=np.int32)
    ds = SurfaceDataSource(
        time=TimeAxis(initial_time=0.0, timestep_size=0.0, n_timesteps=0),
        topology=Topology.triangles(tris, verts),
        elements=ElementMeta(),
        fields=MemoryFieldStore({"pressure": np.array([1.0])}),
    )
    with pytest.raises(ValueError):
        compute_statistics(ds, StatisticsParams(kinds=["mean"]))
