"""Vortex-shedding cross-check on a building's across-wind load record.

A dimensional sanity check that needs no configuration file to agree with: a
bluff prismatic body sheds at a Strouhal number that is a property of its
cross-section, roughly ``0.06 - 0.15`` for rectangular plans. So the dominant
peak of the across-wind load spectrum has a *predictable* frequency::

    f_shedding = St * U_H / D

with ``D`` the across-wind face width. Turn that around and the spectrum
implies a Strouhal number:

    St_implied = f_peak * D / U_H

If ``St_implied`` lands outside the physical band, the load record is not
telling the truth about time -- and the usual cause is a mis-scaled time axis,
which nothing else downstream can detect because every array shape, unit and
range stays plausible (see :class:`cfdmod.dynamics.DimensionalData`). This
caught a real deliverable whose record had been compressed 3.8x: its across-wind
peak implied ``St = 0.38``.

The check is cheap and independent of the case config, so run it on every
dynamic stage before trusting the response.
"""

from __future__ import annotations

__all__ = [
    "STROUHAL_RECTANGULAR",
    "STROUHAL_PHYSICAL_RANGE",
    "SheddingCheck",
    "vortex_shedding_frequency",
    "implied_strouhal",
    "spectral_peak",
    "check_vortex_shedding",
]

import numpy as np
import scipy.signal
from pydantic import BaseModel, ConfigDict
from scipy.ndimage import gaussian_filter

from cfdmod.core.data_source import DataSource

# Nominal Strouhal number for a rectangular tall-building plan. Published values
# for rectangular prisms in turbulent boundary-layer flow sit around 0.06-0.15
# depending on side ratio and turbulence; 0.10 is the usual first estimate.
STROUHAL_RECTANGULAR = 0.10

# Outside this band a bluff rectangular section is not what produced the peak.
STROUHAL_PHYSICAL_RANGE = (0.05, 0.25)


class SheddingCheck(BaseModel):
    """Outcome of :func:`check_vortex_shedding`.

    Attributes:
        expected_hz: ``strouhal * u_h / width``.
        observed_hz: Frequency of the dominant peak of the reduced across-wind
            load spectrum ``f * S(f)``.
        implied_strouhal: ``observed_hz * width / u_h`` -- the dimensionless
            number the record actually implies.
        frequency_ratio: ``observed_hz / expected_hz``.
        passed: Whether ``implied_strouhal`` is inside ``physical_range``.
        physical_range: The band tested against.
    """

    model_config = ConfigDict(frozen=True)

    expected_hz: float
    observed_hz: float
    implied_strouhal: float
    frequency_ratio: float
    passed: bool
    physical_range: tuple[float, float]

    def summary(self) -> str:
        verdict = "OK" if self.passed else "OUT OF RANGE"
        return (
            f"vortex shedding: expected {self.expected_hz:.4f} Hz, observed "
            f"{self.observed_hz:.4f} Hz ({self.frequency_ratio:.2f}x) -> implied "
            f"St = {self.implied_strouhal:.3f} [{verdict}; physical band "
            f"{self.physical_range[0]:g}-{self.physical_range[1]:g}]"
        )

    def raise_if_failed(self) -> "SheddingCheck":
        """Turn the check into an assertion, for a notebook that must not proceed."""
        if not self.passed:
            raise ValueError(
                self.summary() + ". A record whose implied Strouhal number is not "
                "physical usually has a mis-scaled time axis; check the "
                "characteristic length the time axis was de-normalised with."
            )
        return self


def vortex_shedding_frequency(
    u_h: float, across_wind_width: float, *, strouhal: float = STROUHAL_RECTANGULAR
) -> float:
    """``St * U_H / D`` [Hz].

    Args:
        u_h: Reference wind speed at building height [m/s].
        across_wind_width: Width of the face normal to the wind [m] -- the
            dimension the wake spans, not necessarily the plan ``base``.
        strouhal: Strouhal number; defaults to the rectangular-plan estimate.
    """
    if u_h <= 0 or across_wind_width <= 0:
        raise ValueError(
            f"u_h and across_wind_width must be positive; got {u_h!r}, {across_wind_width!r}"
        )
    return float(strouhal) * float(u_h) / float(across_wind_width)


def implied_strouhal(peak_hz: float, u_h: float, across_wind_width: float) -> float:
    """``f_peak * D / U_H`` -- the Strouhal number a measured peak implies."""
    if u_h <= 0:
        raise ValueError(f"u_h must be positive; got {u_h!r}")
    return float(peak_hz) * float(across_wind_width) / float(u_h)


def _reduced_spectrum(
    series: np.ndarray, dt: float, sigma: float
) -> tuple[np.ndarray, np.ndarray]:
    """``(freq, f * S(f) / var)`` -- the reduced spectrum the deliverable plots."""
    freq, psd = scipy.signal.periodogram(series, 1.0 / dt, scaling="density")
    var = float(np.var(series))
    if var <= 0:
        raise ValueError("load series has zero variance; nothing to find a peak in")
    return freq, gaussian_filter(psd * freq / var, sigma=sigma)


def spectral_peak(
    load_source: DataSource,
    *,
    field: str = "cf_y",
    band: tuple[float, float] | None = None,
    sigma: float = 2.0,
    min_cycles: float = 5.0,
) -> float:
    """Frequency [Hz] of the dominant peak of the reduced global-load spectrum.

    The load is summed over floors first (the global across-wind force is what
    sheds), then reduced to ``f * S(f)``, whose maximum is the shedding peak for
    a bluff body.

    ``band`` restricts the search; by default it spans everything the record can
    actually resolve -- from ``min_cycles`` cycles over the record length up to
    Nyquist. Leaving it open is deliberate: narrowing the search around the
    expected frequency would let the check confirm itself.
    """
    arr = np.asarray(load_source.fields.read(field), dtype=np.float64)
    series = arr.sum(axis=0) if arr.ndim == 2 else arr
    dt = float(load_source.time.timestep_size)
    freq, reduced = _reduced_spectrum(series, dt, sigma)

    lo, hi = band if band is not None else (min_cycles / (len(series) * dt), 0.5 / dt)
    mask = (freq >= lo) & (freq <= hi)
    if not mask.any():
        raise ValueError(f"no spectral bins inside band {lo:g}-{hi:g} Hz")
    return float(freq[mask][np.argmax(reduced[mask])])


def check_vortex_shedding(
    load_source: DataSource,
    *,
    u_h: float,
    across_wind_width: float,
    field: str = "cf_y",
    strouhal: float = STROUHAL_RECTANGULAR,
    physical_range: tuple[float, float] = STROUHAL_PHYSICAL_RANGE,
    band: tuple[float, float] | None = None,
    sigma: float = 2.0,
) -> SheddingCheck:
    """Cross-check a load record's time axis against vortex-shedding physics.

    Args:
        load_source: Floor-load source with a **physical** time axis (seconds).
        u_h: Reference wind speed the loads are referenced at [m/s].
        across_wind_width: Width of the face normal to this wind direction [m].
        field: Across-wind load field. ``"cf_y"`` for wind along x.
        strouhal / physical_range: The nominal value and the acceptance band.
        band / sigma: Passed to :func:`spectral_peak`.

    Returns:
        A :class:`SheddingCheck`. Call ``.summary()`` to report it or
        ``.raise_if_failed()`` to stop a notebook that must not continue.
    """
    expected = vortex_shedding_frequency(u_h, across_wind_width, strouhal=strouhal)
    observed = spectral_peak(load_source, field=field, band=band, sigma=sigma)
    st = implied_strouhal(observed, u_h, across_wind_width)
    return SheddingCheck(
        expected_hz=expected,
        observed_hz=observed,
        implied_strouhal=st,
        frequency_ratio=observed / expected,
        passed=bool(physical_range[0] <= st <= physical_range[1]),
        physical_range=physical_range,
    )
