"""Tests for cfdmod.building.peaks (gust factor + peak methods)."""

from __future__ import annotations

import numpy as np
import pytest

from cfdmod.building import gust_peak_factor, peak_value

pytestmark = pytest.mark.unit


def test_gust_peak_factor_known_value():
    # nu*T = 0.2 * 600 = 120; base = sqrt(2 ln 120) = 3.0936; +0.5772/base
    g = gust_peak_factor(0.2, 600.0)
    assert g == pytest.approx(3.0936 + 0.5772 / 3.0936, rel=1e-3)
    base_only = gust_peak_factor(0.2, 600.0, full=False)
    assert base_only < g
    assert base_only == pytest.approx(3.0936, rel=1e-3)


def test_gust_peak_factor_rejects_small_nu_t():
    with pytest.raises(ValueError):
        gust_peak_factor(0.001, 100.0)  # nu*T = 0.1 < 1


def test_peak_value_max_absolute():
    x = np.array([-5.0, 1.0, 3.0])
    assert peak_value(x, "max") == 5.0
    assert peak_value(x, "max", absolute=False) == 3.0


def test_peak_value_peak_factor():
    rng = np.random.default_rng(0)
    x = rng.normal(0.0, 2.0, size=20000)
    g = gust_peak_factor(0.2, 600.0)
    expected = abs(x.mean()) + g * x.std()
    assert peak_value(x, "peak-factor", f0=0.2) == pytest.approx(expected, rel=1e-9)


def test_peak_value_peak_factor_requires_f0():
    with pytest.raises(ValueError):
        peak_value(np.arange(10.0), "peak-factor")


def test_peak_value_gumbel_monotone_in_fractile():
    rng = np.random.default_rng(1)
    x = rng.gumbel(0.0, 1.0, size=5000)
    low = peak_value(x, "gumbel", non_exceedance=0.5)
    high = peak_value(x, "gumbel", non_exceedance=0.95)
    assert high > low


def test_peak_value_empty_is_nan():
    assert np.isnan(peak_value(np.array([]), "max"))


def test_peak_value_unknown_method():
    with pytest.raises(ValueError):
        peak_value(np.arange(5.0), "nope")  # type: ignore[arg-type]
