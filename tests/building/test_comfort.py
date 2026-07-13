"""Tests for cfdmod.building.comfort (occupant-comfort acceleration limits)."""

from __future__ import annotations

import numpy as np
import pytest

from cfdmod.building import (
    comfort_limit,
    melbourne1992_acceleration_limit,
    milli_g_to_mps2,
    mps2_to_milli_g,
    nbcc_acceleration_limit,
    nbr6123_acceleration_limit,
)

pytestmark = pytest.mark.unit


def test_nbr_residential_reference_point():
    # source: NBR 6123 serviceability, a_lim = 0.01 * 4.08 * f0**-0.445 (m/s^2)
    # at f0 = 0.5 Hz -> 0.01 * 4.08 * 0.5**-0.445 = 0.055541615953
    assert nbr6123_acceleration_limit(0.5) == pytest.approx(0.055541615953, rel=1e-9)


def test_nbr_commercial_reference_point():
    # source: NBR 6123 serviceability, a_lim = 0.01 * 6.12 * f0**-0.445 (m/s^2)
    # at f0 = 0.5 Hz -> 0.01 * 6.12 * 0.5**-0.445 = 0.083312423930
    assert nbr6123_acceleration_limit(0.5, occupancy="commercial") == pytest.approx(
        0.083312423930, rel=1e-9
    )


def test_nbr_commercial_is_residential_scaled_by_coeff_ratio():
    # commercial / residential = 6.12 / 4.08, independent of f0
    res = nbr6123_acceleration_limit(0.3)
    com = nbr6123_acceleration_limit(0.3, occupancy="commercial")
    assert com == pytest.approx(res * (6.12 / 4.08), rel=1e-12)


def test_nbr_decreasing_in_f0():
    f0 = np.array([0.1, 0.2, 0.5, 1.0, 2.0])
    limits = nbr6123_acceleration_limit(f0)
    assert np.all(np.diff(limits) < 0.0)


def test_melbourne_reference_point():
    # source: Melbourne & Palmer (1992), a_lim =
    #   sqrt(2 ln(600 f0)) * (0.68 + ln(R)/5) * exp(-3.65 - 0.41 ln f0) (m/s^2)
    # at f0 = 0.2 Hz, R = 10 yr -> 0.177449030031
    assert melbourne1992_acceleration_limit(0.2, return_period_years=10.0) == pytest.approx(
        0.177449030031, rel=1e-9
    )


def test_melbourne_increases_with_return_period():
    low = melbourne1992_acceleration_limit(0.2, return_period_years=5.0)
    high = melbourne1992_acceleration_limit(0.2, return_period_years=50.0)
    assert high > low


def test_nbcc_flat_limits_exact():
    # source: NBCC occupant-comfort criterion, 10-year return period:
    # residential 15 milli-g, office/commercial 25 milli-g (milli-g * 9.806/1000)
    assert nbcc_acceleration_limit() == pytest.approx(15 * 9.806 / 1000, rel=1e-12)
    assert nbcc_acceleration_limit("commercial") == pytest.approx(25 * 9.806 / 1000, rel=1e-12)


def test_nbcc_milli_g_roundtrip():
    assert mps2_to_milli_g(nbcc_acceleration_limit()) == pytest.approx(15.0, rel=1e-12)
    assert mps2_to_milli_g(nbcc_acceleration_limit("commercial")) == pytest.approx(25.0, rel=1e-12)
    assert milli_g_to_mps2(25.0) == pytest.approx(nbcc_acceleration_limit("commercial"), rel=1e-12)


def test_dispatcher_matches_direct_calls():
    assert comfort_limit(0.3, "nbr") == nbr6123_acceleration_limit(0.3)
    assert comfort_limit(0.3, "nbr", occupancy="commercial") == nbr6123_acceleration_limit(
        0.3, occupancy="commercial"
    )
    assert comfort_limit(
        0.2, "melbourne", return_period_years=20.0
    ) == melbourne1992_acceleration_limit(0.2, return_period_years=20.0)
    assert comfort_limit(0.5, "nbcc") == nbcc_acceleration_limit()


def test_vectorization_matches_elementwise():
    f0 = np.array([0.15, 0.3, 0.6, 1.2])
    nbr = nbr6123_acceleration_limit(f0)
    melb = melbourne1992_acceleration_limit(f0, return_period_years=10.0)
    assert isinstance(nbr, np.ndarray)
    assert isinstance(melb, np.ndarray)
    for i, f in enumerate(f0):
        assert nbr[i] == pytest.approx(nbr6123_acceleration_limit(float(f)), rel=1e-12)
        assert melb[i] == pytest.approx(
            melbourne1992_acceleration_limit(float(f), return_period_years=10.0), rel=1e-12
        )


def test_scalar_input_returns_float():
    assert isinstance(nbr6123_acceleration_limit(0.5), float)
    assert isinstance(melbourne1992_acceleration_limit(0.5), float)


def test_guards_raise():
    with pytest.raises(ValueError):
        nbr6123_acceleration_limit(0.0)
    with pytest.raises(ValueError):
        nbr6123_acceleration_limit(-1.0)
    with pytest.raises(ValueError):
        nbr6123_acceleration_limit(0.5, occupancy="hotel")  # type: ignore[arg-type]
    with pytest.raises(ValueError):
        melbourne1992_acceleration_limit(0.5, return_period_years=0.0)
    with pytest.raises(ValueError):
        # 600 * f0 = 0.06 < 1 -> sqrt term undefined
        melbourne1992_acceleration_limit(0.0001)
    with pytest.raises(ValueError):
        nbcc_acceleration_limit("hotel")  # type: ignore[arg-type]
    with pytest.raises(ValueError):
        comfort_limit(0.5, "eurocode")  # type: ignore[arg-type]
