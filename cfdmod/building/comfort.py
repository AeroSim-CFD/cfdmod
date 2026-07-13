"""Wind-induced occupant-comfort acceleration limits.

Tall buildings sway under wind; occupant comfort is judged by comparing the
peak horizontal acceleration at a floor against a standard's acceptance limit.
Three criteria are supported:

- ``"nbr"`` -- ABNT NBR 6123 serviceability limit, a decreasing power law in the
  fundamental sway frequency ``f0`` with separate residential / commercial
  coefficients.
- ``"melbourne"`` -- Melbourne & Palmer (1992) serviceable-acceleration curve,
  a function of ``f0`` and the return period in years.
- ``"nbcc"`` -- NBCC occupant-comfort criterion, a flat (frequency-independent)
  limit at a 10-year return period: 15 milli-g residential, 25 milli-g office /
  commercial.

All functions return the limit in SI ``m/s^2``; converting to milli-g is left to
callers (use :func:`mps2_to_milli_g`). ``f0`` may be a scalar or an
``np.ndarray``; the return matches the input shape, so a limit curve can be
evaluated over a frequency sweep in one call.
"""

from __future__ import annotations

from typing import Literal

import numpy as np

Occupancy = Literal["residential", "commercial"]
Standard = Literal["nbr", "melbourne", "nbcc"]

# Standard-of-record constants, VERIFIED against the primary sources:
# - NBR 6123 (Projeto NBR 6123, sec. 9.6.2 "Aceleracao limite para garantia do
#   conforto humano"): a_lim = 0.01 * k_c * f^-0.445 in m/s^2, with k_c = 4.08
#   (residential) / 6.12 (commercial and office). Valid over 0.06-1.00 Hz, at a
#   1-year return period. The perception curve is ISO 2631-2 / ISO 10137 Annex D
#   (residential ~8x, commercial ~12x the lower perception threshold).
# - Melbourne & Palmer (1992), "Accelerations and comfort criteria for buildings
#   undergoing complex motions", Eq. 3: a = sqrt(2 ln(n0 T)) * (0.68 + ln(R)/5)
#   * exp(-3.65 - 0.41 ln(n0)), with T = 600 s, in m/s^2. Valid over
#   0.06 < n0 < 1.0 Hz and 0.5 < R < 10 years.
# - NBCC: 15 / 25 milli-g (residential / office) at a 10-year return period.
# All three also match AeroSim's production hfpi_analysis notebook.
_NBR_C_RESIDENTIAL = 4.08
_NBR_C_COMMERCIAL = 6.12
_NBR_EXP = -0.445
_CM_TO_M = 0.01

_MELB_AVG_DURATION_S = 600.0
_MELB_A = 0.68
_MELB_B = 5.0
_MELB_C0 = -3.65
_MELB_C1 = -0.41

# NBCC flat occupant-comfort limits (milli-g, 10-year return period):
# residential 15, office / commercial 25.
_NBCC_MG_RESIDENTIAL = 15.0
_NBCC_MG_COMMERCIAL = 25.0

# Standard gravity used for milli-g <-> m/s^2 conversion (m/s^2).
_G = 9.806


def milli_g_to_mps2(milli_g: float | np.ndarray) -> float | np.ndarray:
    """Convert an acceleration from milli-g to ``m/s^2``."""
    return milli_g * _G / 1000.0


def mps2_to_milli_g(mps2: float | np.ndarray) -> float | np.ndarray:
    """Convert an acceleration from ``m/s^2`` to milli-g."""
    return mps2 * 1000.0 / _G


def _require_positive_f0(f0: float | np.ndarray) -> np.ndarray:
    arr = np.asarray(f0, dtype=np.float64)
    if np.any(arr <= 0.0):
        raise ValueError(f"f0 must be positive (got {f0!r})")
    return arr


def _match_shape(f0: float | np.ndarray, value: np.ndarray) -> float | np.ndarray:
    """Return a Python float for scalar input, else the array itself."""
    if np.ndim(f0) == 0:
        return float(value)
    return value


def nbr6123_acceleration_limit(
    f0: float | np.ndarray,
    occupancy: Occupancy = "residential",
) -> float | np.ndarray:
    """ABNT NBR 6123 serviceability acceleration limit (sec. 9.6.2).

    ``a_lim = 0.01 * coeff * f0**-0.445`` with ``coeff`` 4.08 (residential) or
    6.12 (commercial); ``f0`` the fundamental sway frequency in Hz. The ``0.01``
    converts the standard's cm/s^2 expression to m/s^2. Returns m/s^2. The
    standard states this over 0.06-1.00 Hz at a 1-year return period.
    """
    arr = _require_positive_f0(f0)
    if occupancy == "residential":
        coeff = _NBR_C_RESIDENTIAL
    elif occupancy == "commercial":
        coeff = _NBR_C_COMMERCIAL
    else:
        raise ValueError(f"unknown occupancy {occupancy!r}")
    value = _CM_TO_M * coeff * arr**_NBR_EXP
    return _match_shape(f0, value)


def melbourne1992_acceleration_limit(
    f0: float | np.ndarray,
    return_period_years: float = 10.0,
) -> float | np.ndarray:
    """Melbourne & Palmer (1992) serviceable peak-acceleration limit (Eq. 3).

    ``a_lim = sqrt(2 ln(600 f0)) * (0.68 + ln(R) / 5) * exp(-3.65 - 0.41 ln f0)``
    with ``f0`` the fundamental sway frequency (Hz), ``R`` the return period in
    years and ``600`` the averaging window (s). Returns m/s^2. The paper states
    this over 0.06 < f0 < 1.0 Hz and 0.5 < R < 10 years.
    """
    arr = _require_positive_f0(f0)
    if return_period_years <= 0.0:
        raise ValueError(f"return_period_years must be positive (got {return_period_years})")
    if np.any(_MELB_AVG_DURATION_S * arr <= 1.0):
        raise ValueError(f"600 * f0 must exceed 1 for the sqrt term (got f0={f0!r})")
    value = (
        np.sqrt(2.0 * np.log(_MELB_AVG_DURATION_S * arr))
        * (_MELB_A + np.log(return_period_years) / _MELB_B)
        * np.exp(_MELB_C0 + _MELB_C1 * np.log(arr))
    )
    return _match_shape(f0, value)


def nbcc_acceleration_limit(occupancy: Occupancy = "residential") -> float:
    """NBCC flat occupant-comfort limit (10-year return period), in m/s^2.

    Frequency-independent: 15 milli-g residential, 25 milli-g office /
    commercial.
    """
    if occupancy == "residential":
        milli_g = _NBCC_MG_RESIDENTIAL
    elif occupancy == "commercial":
        milli_g = _NBCC_MG_COMMERCIAL
    else:
        raise ValueError(f"unknown occupancy {occupancy!r}")
    return float(milli_g_to_mps2(milli_g))


def comfort_limit(
    f0: float | np.ndarray,
    standard: Standard,
    *,
    occupancy: Occupancy = "residential",
    return_period_years: float = 10.0,
) -> float | np.ndarray:
    """Dispatch to an acceleration-limit curve by ``standard``.

    Args:
        f0: fundamental sway frequency (Hz); scalar or array.
        standard: ``"nbr"``, ``"melbourne"`` or ``"nbcc"``.
        occupancy: ``"residential"`` or ``"commercial"`` (NBR / NBCC).
        return_period_years: return period (Melbourne only).

    Returns the limit in m/s^2 (matching the shape of ``f0``, except the flat
    NBCC limit which is a scalar).
    """
    if standard == "nbr":
        return nbr6123_acceleration_limit(f0, occupancy=occupancy)
    if standard == "melbourne":
        return melbourne1992_acceleration_limit(f0, return_period_years=return_period_years)
    if standard == "nbcc":
        return nbcc_acceleration_limit(occupancy=occupancy)
    raise ValueError(f"unknown standard {standard!r}")
