"""Tests for the pure-numpy filter core (cfdmod.filters.apply_filters)."""

from __future__ import annotations

import numpy as np
import pytest

from cfdmod.filters import MovingAverageFilter, apply_filters
from cfdmod.filters.core import _window_in_samples

pytestmark = pytest.mark.unit


def test_window_in_samples_rounds_to_odd():
    assert _window_in_samples(window=1.0, dt=0.1) == 11
    assert _window_in_samples(window=0.01, dt=0.1) == 1


def test_apply_filters_2d_smooths_along_axis_0():
    rng = np.random.default_rng(0)
    n_t, n_feat = 200, 3
    t = np.linspace(0.0, 10.0, n_t)
    signal = np.sin(2 * np.pi * t)[:, None] + rng.normal(0.0, 0.5, (n_t, n_feat))

    out = apply_filters(signal, dt=t[1] - t[0], filters=[MovingAverageFilter(window=1.0)])
    assert out.shape == signal.shape
    assert out.std() < 0.5 * signal.std()


def test_apply_filters_1d_input_returns_1d():
    n_t = 100
    t = np.linspace(0.0, 5.0, n_t)
    signal = np.sin(2 * np.pi * t) + 0.5 * np.random.default_rng(1).normal(size=n_t)

    out = apply_filters(signal, dt=t[1] - t[0], filters=[MovingAverageFilter(window=1.0)])
    assert out.shape == signal.shape
    assert out.ndim == 1
    assert out.std() < signal.std()


def test_apply_filters_preserves_input_dtype():
    n_t = 50
    signal = np.ones((n_t, 2), dtype=np.float32)
    out = apply_filters(signal, dt=0.1, filters=[MovingAverageFilter(window=0.5)])
    assert out.dtype == np.float32


def test_apply_filters_chain_order_matters():
    rng = np.random.default_rng(2)
    n_t, n_feat = 200, 4
    t = np.linspace(0.0, 10.0, n_t)
    signal = np.sin(2 * np.pi * t)[:, None] + rng.normal(0.0, 0.5, (n_t, n_feat))
    dt = float(t[1] - t[0])

    one = apply_filters(signal, dt, [MovingAverageFilter(window=0.5)])
    two = apply_filters(
        signal, dt, [MovingAverageFilter(window=0.5), MovingAverageFilter(window=0.5)]
    )
    assert two.std() < one.std()


def test_apply_filters_constant_signal_unchanged():
    n_t = 100
    signal = np.full((n_t, 3), 7.5, dtype=np.float64)
    out = apply_filters(signal, dt=0.1, filters=[MovingAverageFilter(window=1.0)])
    np.testing.assert_allclose(out, signal)


def test_apply_filters_window_below_dt_is_noop():
    """A window so small it rounds to 1 sample short-circuits."""
    rng = np.random.default_rng(3)
    signal = rng.normal(size=(50, 2))
    out = apply_filters(signal, dt=0.1, filters=[MovingAverageFilter(window=0.001)])
    np.testing.assert_array_equal(out, signal)


def test_apply_filters_empty_chain_raises():
    signal = np.zeros((10, 2))
    with pytest.raises(ValueError, match="filters list is empty"):
        apply_filters(signal, dt=0.1, filters=[])


def test_apply_filters_non_positive_dt_raises():
    signal = np.zeros((10, 2))
    with pytest.raises(ValueError, match="dt must be > 0"):
        apply_filters(signal, dt=0.0, filters=[MovingAverageFilter(window=1.0)])
    with pytest.raises(ValueError, match="dt must be > 0"):
        apply_filters(signal, dt=-0.1, filters=[MovingAverageFilter(window=1.0)])


def test_apply_filters_too_short_axis_raises():
    signal = np.zeros((1, 3))
    with pytest.raises(ValueError, match="at least 2 timesteps"):
        apply_filters(signal, dt=0.1, filters=[MovingAverageFilter(window=1.0)])


def test_apply_filters_invalid_ndim_raises():
    signal = np.zeros((4, 4, 4))
    with pytest.raises(ValueError, match="must be 1D or 2D"):
        apply_filters(signal, dt=0.1, filters=[MovingAverageFilter(window=1.0)])


def test_top_level_imports_are_reachable():
    from cfdmod import MovingAverageFilter as top_ma
    from cfdmod import apply_filters as top_apply
    from cfdmod import apply_filters_h5 as top_apply_h5
    from cfdmod.filters import apply_filters as core_apply
    from cfdmod.filters import apply_filters_h5 as core_apply_h5

    assert top_apply is core_apply
    assert top_apply_h5 is core_apply_h5
    assert top_ma is MovingAverageFilter
