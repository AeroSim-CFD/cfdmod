"""Tests for the pure-numpy statistics core (cfdmod.statistics.apply_statistics)."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from cfdmod.statistics import (
    BasicStatisticModel,
    ExtremeAbsoluteParamsModel,
    ExtremePeakParamsModel,
    ParameterizedStatisticModel,
    apply_statistics,
)

pytestmark = pytest.mark.unit


def _signal(n_t=200, n_feat=4, seed=0):
    rng = np.random.default_rng(seed)
    t = np.linspace(0.0, 10.0, n_t)
    sig = np.sin(2 * np.pi * t)[:, None] + rng.normal(0.0, 0.5, (n_t, n_feat))
    return t, sig


def test_basic_moments_match_numpy_directly():
    t, sig = _signal()
    stats = [
        BasicStatisticModel(stats="mean"),
        BasicStatisticModel(stats="rms"),
    ]
    df = apply_statistics(sig, time=t, statistics=stats)
    assert isinstance(df, pd.DataFrame)
    assert set(df.columns) == {"mean", "rms"}
    np.testing.assert_allclose(df["mean"].values, sig.mean(axis=0), rtol=1e-10)
    np.testing.assert_allclose(df["rms"].values, sig.std(axis=0, ddof=1), rtol=1e-10)


def test_min_max_with_absolute_method():
    t, sig = _signal()
    stats = [
        ParameterizedStatisticModel(stats="min", params=ExtremeAbsoluteParamsModel()),
        ParameterizedStatisticModel(stats="max", params=ExtremeAbsoluteParamsModel()),
    ]
    df = apply_statistics(sig, time=t, statistics=stats)
    np.testing.assert_allclose(df["min"].values, sig.min(axis=0))
    np.testing.assert_allclose(df["max"].values, sig.max(axis=0))


def test_peak_method_uses_factor():
    t, sig = _signal(n_feat=2)
    stats = [
        ParameterizedStatisticModel(
            stats="min", params=ExtremePeakParamsModel(peak_factor=2.0)
        ),
        ParameterizedStatisticModel(
            stats="max", params=ExtremePeakParamsModel(peak_factor=2.0)
        ),
    ]
    df = apply_statistics(sig, time=t, statistics=stats)
    # peak_extreme_values delegates to pandas Series.std() (ddof=1).
    expected_max = sig.mean(axis=0) + 2.0 * sig.std(axis=0, ddof=1)
    expected_min = sig.mean(axis=0) - 2.0 * sig.std(axis=0, ddof=1)
    np.testing.assert_allclose(df["max"].values, expected_max, rtol=1e-12)
    np.testing.assert_allclose(df["min"].values, expected_min, rtol=1e-12)


def test_1d_input_returns_single_row_frame():
    t, sig = _signal(n_feat=1)
    df = apply_statistics(
        sig[:, 0], time=t, statistics=[BasicStatisticModel(stats="mean")]
    )
    assert df.shape == (1, 1)
    assert "mean" in df.columns
    assert df["mean"].iloc[0] == pytest.approx(sig[:, 0].mean())


def test_empty_statistics_list_raises():
    t, sig = _signal()
    with pytest.raises(ValueError, match="statistics list is empty"):
        apply_statistics(sig, time=t, statistics=[])


def test_time_length_must_match_data_axis():
    _, sig = _signal()
    bad_time = np.linspace(0.0, 1.0, sig.shape[0] + 1)
    with pytest.raises(ValueError, match="time must be 1D with length"):
        apply_statistics(
            sig, time=bad_time, statistics=[BasicStatisticModel(stats="mean")]
        )


def test_invalid_ndim_raises():
    sig = np.zeros((4, 4, 4))
    t = np.arange(4)
    with pytest.raises(ValueError, match="must be 1D or 2D"):
        apply_statistics(sig, time=t, statistics=[BasicStatisticModel(stats="mean")])


def test_top_level_imports_are_reachable():
    from cfdmod import apply_statistics as top_apply
    from cfdmod import apply_statistics_h5 as top_apply_h5
    from cfdmod.statistics import apply_statistics as core_apply
    from cfdmod.statistics import apply_statistics_h5 as core_apply_h5

    assert top_apply is core_apply
    assert top_apply_h5 is core_apply_h5
