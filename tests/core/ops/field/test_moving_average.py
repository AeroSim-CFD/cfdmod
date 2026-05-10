"""Unit tests for the moving-average field op.

Equivalence with the legacy ``cfdmod.pressure.filters._apply_one``
implementation is asserted on a synthetic series so any future change
to the rounding rule or padding mode shows up.
"""

from __future__ import annotations

import numpy as np
import pytest

from cfdmod.adapters.memory import MemoryFieldStore
from cfdmod.core import ElementMeta, SurfaceDataSource, TimeAxis, Topology
from cfdmod.core.ops.field.moving_average import (
    MovingAverageParams,
    moving_average,
    window_in_samples,
)
from cfdmod.pressure.filters import MovingAverageFilter, _apply_one


def _surface(values: np.ndarray, dt: float = 0.1) -> SurfaceDataSource:
    n_elements, n_timesteps = values.shape
    verts = np.tile(np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0]]), (n_elements, 1)).astype(
        np.float64
    )
    tris = (np.arange(n_elements * 3).reshape(n_elements, 3)).astype(np.int32)
    return SurfaceDataSource(
        time=TimeAxis(initial_time=0.0, timestep_size=dt, n_timesteps=n_timesteps),
        topology=Topology.triangles(tris, verts),
        elements=ElementMeta(),
        fields=MemoryFieldStore({"pressure": values.astype(np.float64)}),
    )


def test_window_in_samples_rounds_to_odd():
    assert window_in_samples(0.5, 0.1) == 5
    assert window_in_samples(0.6, 0.1) == 7  # 6 -> next odd
    assert window_in_samples(0.001, 0.1) == 1  # below 1 sample -> 1


def test_moving_average_window_one_is_identity():
    rng = np.random.default_rng(0)
    data = rng.normal(size=(3, 20))
    ds = _surface(data)
    out = moving_average(ds, MovingAverageParams(window=0.001))  # rounds to 1 sample
    np.testing.assert_array_equal(out.fields.read("pressure"), data)


def test_moving_average_matches_legacy():
    """v3 op must produce the same numbers as the legacy _apply_one."""
    rng = np.random.default_rng(42)
    n_elements, n_timesteps = 4, 50
    data = rng.normal(size=(n_elements, n_timesteps))
    dt = 0.05
    ds = _surface(data, dt=dt)

    out = moving_average(ds, MovingAverageParams(window=0.5))

    # Legacy operates on (n_time, n_tri); transpose for the comparison.
    legacy_in = data.T.copy()
    legacy_out = _apply_one(MovingAverageFilter(window=0.5), legacy_in, dt)
    np.testing.assert_allclose(out.fields.read("pressure"), legacy_out.T, rtol=1e-12)


def test_moving_average_smooths_step_function():
    data = np.zeros((1, 20), dtype=np.float64)
    data[0, 10:] = 1.0
    ds = _surface(data)
    out = moving_average(ds, MovingAverageParams(window=0.5))
    smoothed = out.fields.read("pressure")[0]
    assert smoothed[0] == pytest.approx(0.0)
    assert smoothed[-1] == pytest.approx(1.0)
    # Monotonic non-decreasing through the transition.
    assert np.all(np.diff(smoothed) >= -1e-12)


def test_moving_average_writes_to_out_field():
    data = np.arange(10.0).reshape(1, 10)
    ds = _surface(data)
    out = moving_average(ds, MovingAverageParams(window=0.3, out="pressure_smooth"))
    assert sorted(out.field_names) == ["pressure", "pressure_smooth"]
    np.testing.assert_array_equal(out.fields.read("pressure"), data)


def test_moving_average_rejects_time_aggregated_source():
    verts = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0]], dtype=np.float64)
    tris = np.array([[0, 1, 2]], dtype=np.int32)
    ds = SurfaceDataSource(
        time=TimeAxis(initial_time=0.0, timestep_size=0.0, n_timesteps=0),
        topology=Topology.triangles(tris, verts),
        elements=ElementMeta(),
        fields=MemoryFieldStore({"pressure": np.array([1.0])}),
    )
    with pytest.raises(ValueError):
        moving_average(ds, MovingAverageParams(window=0.1))
