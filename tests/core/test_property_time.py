"""Property-based time-op tests (issue #141 phase 3).

Covers the affine :class:`~cfdmod.core.time_axis.TimeAxis` primitives and the
centred moving-average field op:

- ``window(start, end)`` returns a contiguous slice, and the new axis' times
  equal the original times restricted to that slice.
- ``rescale(k)`` then ``rescale(1/k)`` recovers the axis (within tolerance).
- ``moving_average`` with a one-sample window is the identity, always preserves
  shape, and leaves a constant series unchanged.
"""

from __future__ import annotations

import numpy as np
import pytest
from hypothesis import given
from hypothesis import strategies as st
from hypothesis.extra import numpy as hnp

from cfdmod.adapters.memory import MemoryFieldStore
from cfdmod.core.data_source import PointsDataSource
from cfdmod.core.ops.field.moving_average import MovingAverageParams, moving_average
from cfdmod.core.time_axis import TimeAxis
from cfdmod.core.topology import ElementMeta, Topology
from tests import strategies as sty

pytestmark = pytest.mark.property


def _finite(**kw):
    return st.floats(allow_nan=False, allow_infinity=False, width=64, **kw)


# ---------------------------------------------------------------------------
# TimeAxis.window: contiguous slice, times restricted to the slice.
# ---------------------------------------------------------------------------


@given(axis=sty.time_axes(aggregated=False), data=st.data())
def test_window_is_contiguous_slice(axis: TimeAxis, data) -> None:
    times = axis.times()
    first, last = float(times[0]), float(times[-1])
    lo = data.draw(_finite(min_value=first, max_value=last))
    hi = data.draw(_finite(min_value=first, max_value=last))
    start, end = min(lo, hi), max(lo, hi)

    new_axis, idx = axis.window(start, end)
    # Contiguous: a plain slice with unit step.
    assert idx.step in (None, 1)
    assert 0 <= idx.start < idx.stop <= axis.n_timesteps
    assert new_axis.n_timesteps == idx.stop - idx.start
    assert np.allclose(new_axis.times(), times[idx])


# ---------------------------------------------------------------------------
# TimeAxis.rescale: reversible.
# ---------------------------------------------------------------------------


@given(axis=sty.time_axes(), k=_finite(min_value=1e-3, max_value=1e3))
def test_rescale_roundtrip_recovers_axis(axis: TimeAxis, k: float) -> None:
    back = axis.rescale(k).rescale(1.0 / k)
    assert np.isclose(back.initial_time, axis.initial_time, rtol=1e-9, atol=1e-9)
    assert np.isclose(back.timestep_size, axis.timestep_size, rtol=1e-9, atol=1e-12)
    assert back.n_timesteps == axis.n_timesteps


# ---------------------------------------------------------------------------
# moving_average.
# ---------------------------------------------------------------------------


def _timeseries(arr: np.ndarray, dt: float) -> PointsDataSource:
    n = arr.shape[0]
    return PointsDataSource(
        time=TimeAxis(initial_time=0.0, timestep_size=dt, n_timesteps=arr.shape[1]),
        topology=Topology.points(np.zeros((n, 3), dtype=np.float64)),
        elements=ElementMeta(),
        fields=MemoryFieldStore({"v": arr}),
    )


@st.composite
def _timeseries_source(draw):
    n = draw(st.integers(min_value=1, max_value=5))
    nt = draw(st.integers(min_value=2, max_value=8))
    dt = draw(_finite(min_value=1e-2, max_value=10.0))
    arr = draw(hnp.arrays(np.float64, (n, nt), elements=_finite(min_value=-1e4, max_value=1e4)))
    return _timeseries(arr, dt)


@given(ds=_timeseries_source())
def test_moving_average_one_sample_is_identity(ds: PointsDataSource) -> None:
    # window == dt rounds to a single sample -> identity.
    dt = ds.time.timestep_size
    out = moving_average(ds, MovingAverageParams(window=dt, field="v"))
    assert np.array_equal(out.fields.read("v"), ds.fields.read("v"))


@given(ds=_timeseries_source(), window=_finite(min_value=1e-2, max_value=50.0))
def test_moving_average_preserves_shape(ds: PointsDataSource, window: float) -> None:
    out = moving_average(ds, MovingAverageParams(window=window, field="v"))
    assert out.fields.read("v").shape == ds.fields.read("v").shape


@given(
    n=st.integers(min_value=1, max_value=4),
    nt=st.integers(min_value=2, max_value=8),
    dt=_finite(min_value=1e-2, max_value=10.0),
    c=_finite(min_value=-1e4, max_value=1e4),
    window=_finite(min_value=1e-2, max_value=50.0),
)
def test_moving_average_of_constant_is_constant(n, nt, dt, c, window) -> None:
    ds = _timeseries(np.full((n, nt), c, dtype=np.float64), dt)
    out = moving_average(ds, MovingAverageParams(window=window, field="v"))
    assert np.allclose(out.fields.read("v"), c)
