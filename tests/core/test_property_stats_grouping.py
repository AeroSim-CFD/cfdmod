"""Property-based statistics and grouping-aggregation tests (issue #141 phase 2).

Two computational cores are fuzzed against a numpy reference and their own
ordering invariants:

- :func:`cfdmod.core.grouping.aggregate_rows` -- ``min <= mean <= max``,
  ``area_weighted_mean == mean`` for equal weights, and ``sum == mean * n`` for
  a full group.
- :func:`cfdmod.core.ops.data_source_create.statistics.compute_statistics` --
  ``mean``/``min``/``max``/``rms`` match numpy on random arrays, and
  ``min <= mean <= max``.

All arrays are finite by default; sum/mean identities are compared with a
floating-point tolerance rather than exact equality.
"""

from __future__ import annotations

import numpy as np
import pytest
from hypothesis import given
from hypothesis import strategies as st
from hypothesis.extra import numpy as hnp

from cfdmod.adapters.memory import MemoryFieldStore
from cfdmod.core.data_source import PointsDataSource
from cfdmod.core.grouping import aggregate_rows
from cfdmod.core.ops.data_source_create.statistics import (
    StatisticsParams,
    compute_statistics,
)
from cfdmod.core.time_axis import TimeAxis
from cfdmod.core.topology import ElementMeta, Topology

pytestmark = pytest.mark.property


def _finite(**kw):
    return st.floats(allow_nan=False, allow_infinity=False, width=64, **kw)


@st.composite
def _row_arrays(draw):
    """A 2-D finite array (n_rows, n_cols) plus a non-empty member index set."""
    n_rows = draw(st.integers(min_value=1, max_value=8))
    n_cols = draw(st.integers(min_value=1, max_value=4))
    arr = draw(
        hnp.arrays(np.float64, (n_rows, n_cols), elements=_finite(min_value=-1e6, max_value=1e6))
    )
    members = draw(
        st.lists(
            st.integers(min_value=0, max_value=n_rows - 1),
            min_size=1,
            max_size=n_rows,
            unique=True,
        )
    )
    return arr, np.array(sorted(members), dtype=np.int64)


@given(payload=_row_arrays())
def test_aggregate_min_le_mean_le_max(payload) -> None:
    arr, members = payload
    mn = aggregate_rows(arr, members, "min")
    me = aggregate_rows(arr, members, "mean")
    mx = aggregate_rows(arr, members, "max")
    assert np.all(mn <= me + 1e-9)
    assert np.all(me <= mx + 1e-9)


@given(payload=_row_arrays(), weight=_finite(min_value=1e-3, max_value=1e3))
def test_area_weighted_mean_equals_mean_for_equal_weights(payload, weight: float) -> None:
    arr, members = payload
    weights = np.full(arr.shape[0], weight, dtype=np.float64)
    aw = aggregate_rows(arr, members, "area_weighted_mean", weights=weights)
    me = aggregate_rows(arr, members, "mean")
    assert np.allclose(aw, me, rtol=1e-9, atol=1e-9)


@given(payload=_row_arrays())
def test_sum_equals_mean_times_n_for_full_group(payload) -> None:
    arr, _members = payload
    full = np.arange(arr.shape[0], dtype=np.int64)
    total = aggregate_rows(arr, full, "sum")
    mean = aggregate_rows(arr, full, "mean")
    assert np.allclose(total, mean * arr.shape[0], rtol=1e-9, atol=1e-6)


def _points_timeseries(arr: np.ndarray) -> PointsDataSource:
    n = arr.shape[0]
    return PointsDataSource(
        time=TimeAxis(initial_time=0.0, timestep_size=1.0, n_timesteps=arr.shape[1]),
        topology=Topology.points(np.zeros((n, 3), dtype=np.float64)),
        elements=ElementMeta(),
        fields=MemoryFieldStore({"p": arr}),
    )


@st.composite
def _timeseries_arrays(draw):
    n = draw(st.integers(min_value=1, max_value=6))
    nt = draw(st.integers(min_value=2, max_value=6))  # >= 2 so rms (ddof=1) is defined
    return draw(hnp.arrays(np.float64, (n, nt), elements=_finite(min_value=-1e6, max_value=1e6)))


@given(arr=_timeseries_arrays())
def test_statistics_match_numpy(arr: np.ndarray) -> None:
    ds = _points_timeseries(arr)
    out = compute_statistics(ds, StatisticsParams(kinds=["mean", "min", "max", "rms"], field="p"))
    mean = out.fields.read("mean")
    mn = out.fields.read("min")
    mx = out.fields.read("max")
    rms = out.fields.read("rms")
    assert np.allclose(mean, arr.mean(axis=1))
    assert np.allclose(mn, arr.min(axis=1))
    assert np.allclose(mx, arr.max(axis=1))
    assert np.allclose(rms, arr.std(axis=1, ddof=1))
    # Ordering invariant.
    assert np.all(mn <= mean + 1e-9)
    assert np.all(mean <= mx + 1e-9)
