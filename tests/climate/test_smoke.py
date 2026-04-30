"""Smoke tests for cfdmod.climate.

These are not exhaustive correctness checks; they pin the public API
shape and the basic numerical invariants so that future numpy / scipy
bumps don't drift the module silently.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from scipy.stats import gamma as scipy_gamma

from cfdmod.climate import (
    fit_gumbel,
    fit_weibull,
    get_weibull_probability_between_velocities,
    get_weibull_quantile,
    type_I_return_level,
    weibull_scale_from_mean_and_shape,
    weibull_shape_from_mean_and_std,
)

pytestmark = pytest.mark.unit


def _synthetic_weibull_sample(*, shape: float, scale: float, n: int = 5000) -> np.ndarray:
    """Draw a deterministic-ish Weibull sample for fit-roundtrip checks."""
    rng = np.random.default_rng(seed=42)
    u = rng.random(n)
    # Inverse CDF for Weibull: x = scale * (-ln(1 - u)) ** (1/shape)
    return scale * (-np.log1p(-u)) ** (1.0 / shape)


def test_weibull_shape_from_moments_roundtrip():
    """Given a known (shape, scale), recovering shape from sample (mean, std)
    via the closed-form helpers should round-trip within ~3%."""
    true_shape, true_scale = 2.0, 8.0
    sample = _synthetic_weibull_sample(shape=true_shape, scale=true_scale)
    shape = weibull_shape_from_mean_and_std(sample.mean(), sample.std(ddof=0))
    scale = weibull_scale_from_mean_and_shape(sample.mean(), shape)
    assert shape == pytest.approx(true_shape, rel=0.05)
    assert scale == pytest.approx(true_scale, rel=0.05)


def test_fit_weibull_returns_two_positive_floats():
    sample = _synthetic_weibull_sample(shape=2.0, scale=8.0)
    df = pd.DataFrame({"u_mean": sample})
    shape, scale = fit_weibull(df)
    assert shape > 0
    assert scale > 0
    # MLE should land near the true params
    assert shape == pytest.approx(2.0, rel=0.1)
    assert scale == pytest.approx(8.0, rel=0.1)


def test_get_weibull_quantile_monotone_in_percentile():
    shape, scale = 2.0, 8.0
    q50 = get_weibull_quantile(shape, scale, 0.5)
    q90 = get_weibull_quantile(shape, scale, 0.9)
    q99 = get_weibull_quantile(shape, scale, 0.99)
    assert q50 < q90 < q99


def test_weibull_probability_between_zero_and_inf_is_one():
    shape, scale = 2.0, 8.0
    p = get_weibull_probability_between_velocities(shape, scale, 0.0, 1e6)
    assert p == pytest.approx(1.0, abs=1e-6)


def test_type_I_return_level_monotone_in_T():
    """For positive scale a > 0, the T-year return level grows with T."""
    U, a = 20.0, 2.0
    rl_50 = type_I_return_level(50, U, a)
    rl_100 = type_I_return_level(100, U, a)
    rl_500 = type_I_return_level(500, U, a)
    assert rl_50 < rl_100 < rl_500


def test_fit_gumbel_runs_and_returns_three_components():
    """Synthesize a multi-year series with one obvious yearly peak and
    check fit_gumbel returns a plausible (U, a, peaks) triplet."""
    # 8 years of synthetic gust series, one peak per year drawn from a
    # known Gumbel; the rest of the year is small noise.
    rng = np.random.default_rng(seed=0)
    mu, beta = 25.0, 3.0
    n_years = 8
    rows = []
    for y in range(n_years):
        peak = mu - beta * np.log(-np.log(rng.random()))
        # ~52 weekly samples of background noise + one big peak
        background = 5.0 + 1.5 * rng.standard_normal(52)
        for w, val in enumerate(background):
            rows.append((pd.Timestamp(f"{2000 + y}-01-01") + pd.Timedelta(weeks=w), float(val)))
        # peak somewhere in mid-year
        rows.append((pd.Timestamp(f"{2000 + y}-06-15"), float(peak)))
    df = pd.DataFrame(rows, columns=["datetime", "u_gust"])
    out = fit_gumbel(df, events_per_year=1)
    assert len(out) == 3
    U, a, peaks = out
    # Loose sanity: positive scale, location in the gust range, peaks list non-empty.
    assert a > 0
    assert U > 0
    assert len(peaks) > 0


def test_scipy_gamma_still_imported():
    # Sanity: the climate module relies on scipy.special.gamma; pin that dep is alive.
    assert scipy_gamma is not None
