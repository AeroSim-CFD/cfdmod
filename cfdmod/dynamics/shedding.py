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
    "wind_unit_vector",
    "across_wind_width",
    "across_wind_series",
    "strouhal_by_direction",
    "plot_strouhal_by_direction",
]

import warnings

import numpy as np
from pydantic import BaseModel, ConfigDict

from cfdmod.core.data_source import DataSource
from cfdmod.core.recipes.spectral import reduced_spectrum

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


def _warn_sigma(sigma: float | None, name: str) -> None:
    if sigma is not None:
        warnings.warn(
            f"{name}(sigma=...) is deprecated and ignored: the spectrum is a Welch average; "
            "use n_segments",
            DeprecationWarning,
            stacklevel=3,
        )


def _peak_frequency(
    series: np.ndarray,
    dt: float,
    *,
    band: tuple[float, float] | None,
    n_segments: int,
    min_cycles: float,
) -> float:
    """Frequency of the maximum of the Welch reduced spectrum inside ``band``."""
    freq, reduced = reduced_spectrum(series, dt, n_segments=n_segments)
    # freq[0] is the Welch frequency step 1 / T_seg, so the default lower edge is
    # min_cycles cycles per segment: clear of the Hann main lobe around DC (+-2 bins).
    lo, hi = band if band is not None else (min_cycles * float(freq[0]), 0.5 / dt)
    mask = (freq >= lo) & (freq <= hi)
    if not mask.any():
        raise ValueError(f"no spectral bins inside band {lo:g}-{hi:g} Hz")
    return float(freq[mask][np.argmax(reduced[mask])])


def spectral_peak(
    load_source: DataSource,
    *,
    field: str = "cf_y",
    band: tuple[float, float] | None = None,
    n_segments: int = 8,
    min_cycles: float = 3.0,
    sigma: float | None = None,
) -> float:
    """Frequency [Hz] of the dominant peak of the reduced global-load spectrum.

    The load is summed over floors first (the global across-wind force is what
    sheds), then reduced to ``f * S(f)`` (Welch, see
    :func:`cfdmod.core.recipes.spectral.reduced_spectrum`), whose maximum is the
    shedding peak for a bluff body.

    ``band`` restricts the search; by default it spans everything the spectrum can
    actually resolve -- from ``min_cycles`` cycles over one Welch segment up to
    Nyquist. Leaving it open is deliberate: narrowing the search around the
    expected frequency would let the check confirm itself.

    ``sigma`` is deprecated and ignored.
    """
    _warn_sigma(sigma, "spectral_peak")
    arr = np.asarray(load_source.fields.read(field), dtype=np.float64)
    series = arr.sum(axis=0) if arr.ndim == 2 else arr
    dt = float(load_source.time.timestep_size)
    return _peak_frequency(series, dt, band=band, n_segments=n_segments, min_cycles=min_cycles)


def check_vortex_shedding(
    load_source: DataSource,
    *,
    u_h: float,
    across_wind_width: float,
    field: str = "cf_y",
    strouhal: float = STROUHAL_RECTANGULAR,
    physical_range: tuple[float, float] = STROUHAL_PHYSICAL_RANGE,
    band: tuple[float, float] | None = None,
    n_segments: int = 8,
    sigma: float | None = None,
) -> SheddingCheck:
    """Cross-check a load record's time axis against vortex-shedding physics.

    Args:
        load_source: Floor-load source with a **physical** time axis (seconds).
        u_h: Reference wind speed the loads are referenced at [m/s].
        across_wind_width: Width of the face normal to this wind direction [m].
        field: Across-wind load field. ``"cf_y"`` for wind along x.
        strouhal / physical_range: The nominal value and the acceptance band.
        band / n_segments: Passed to :func:`spectral_peak`.
        sigma: Deprecated and ignored.

    Returns:
        A :class:`SheddingCheck`. Call ``.summary()`` to report it or
        ``.raise_if_failed()`` to stop a notebook that must not continue.
    """
    _warn_sigma(sigma, "check_vortex_shedding")
    expected = vortex_shedding_frequency(u_h, across_wind_width, strouhal=strouhal)
    observed = spectral_peak(load_source, field=field, band=band, n_segments=n_segments)
    st = implied_strouhal(observed, u_h, across_wind_width)
    return SheddingCheck(
        expected_hz=expected,
        observed_hz=observed,
        implied_strouhal=st,
        frequency_ratio=observed / expected,
        passed=bool(physical_range[0] <= st <= physical_range[1]),
        physical_range=physical_range,
    )


def wind_unit_vector(direction_deg: float) -> np.ndarray:
    """Unit vector the wind blows *towards*, in the model frame.

    ``0 deg`` blows along ``+x``, ``90 deg`` along ``+y`` -- the convention the
    per-floor ``cf_x`` / ``cf_y`` fields are written in, verified on a real case
    by watching the facade stagnation patch move face to face as the direction
    rotates. Check it on any new case before trusting a directional sweep.
    """
    a = np.deg2rad(float(direction_deg))
    return np.array([np.cos(a), np.sin(a)])


def across_wind_width(footprint_xy, direction_deg: float) -> float:
    """Frontal width [m] of a plan footprint for one wind direction.

    The width the wake spans: the span of the footprint projected onto the axis
    perpendicular to the wind. Exact for any plan shape, so it does not assume a
    rectangle -- for a rectangle ``a`` x ``b`` it reduces to
    ``a*|sin| + b*|cos|``.

    Args:
        footprint_xy: ``(n, 2)`` plan coordinates. Use the *tower* footprint; a
            podium or skirt would inflate the width and deflate the Strouhal
            number.
        direction_deg: Wind direction, see :func:`wind_unit_vector`.
    """
    pts = np.asarray(footprint_xy, dtype=np.float64)
    if pts.ndim != 2 or pts.shape[1] != 2:
        raise ValueError(f"footprint_xy must be (n, 2); got {pts.shape}")
    if pts.shape[0] == 0:
        raise ValueError("footprint_xy is empty")
    d = wind_unit_vector(direction_deg)
    across = pts[:, 0] * -d[1] + pts[:, 1] * d[0]  # projection on the across-wind axis
    return float(across.max() - across.min())


def across_wind_series(
    load_source: DataSource, direction_deg: float, *, field_x: str = "cf_x", field_y: str = "cf_y"
) -> np.ndarray:
    """Global across-wind force history for one direction [same units as input].

    Rotates the model-frame floor loads into the wind frame and sums over
    floors: ``F_across = -Fx*sin(theta) + Fy*cos(theta)``. This is the component
    the wake drives, and the one whose spectrum carries the shedding peak; at an
    oblique direction neither ``cf_x`` nor ``cf_y`` is it on its own.
    """
    d = wind_unit_vector(direction_deg)
    fx = np.asarray(load_source.fields.read(field_x), dtype=np.float64)
    fy = np.asarray(load_source.fields.read(field_y), dtype=np.float64)
    total_x = fx.sum(axis=0) if fx.ndim == 2 else fx
    total_y = fy.sum(axis=0) if fy.ndim == 2 else fy
    return -total_x * d[1] + total_y * d[0]


def strouhal_by_direction(
    load_by_direction: dict[float, DataSource],
    footprint_xy,
    *,
    u_h_by_direction: dict[float, float] | float,
    strouhal: float = STROUHAL_RECTANGULAR,
    physical_range: tuple[float, float] = STROUHAL_PHYSICAL_RANGE,
    n_segments: int = 8,
    min_cycles: float = 3.0,
    sigma: float | None = None,
):
    """Sweep the vortex-shedding check over every wind direction.

    One row per direction: the frontal width, the observed across-wind spectral
    peak, the frequency Strouhal predicts, and the Strouhal number the record
    implies. Read as a *case quality* indicator - the implied number should sit
    in a tight band around the nominal across all directions. A flat offset
    points at the time axis; scatter points at a short record or a plan whose
    shedding is not well defined at oblique angles.

    Args:
        load_by_direction: ``{direction_deg: floor-load source}``, each on a
            physical (seconds) time axis and carrying ``cf_x`` / ``cf_y``.
        footprint_xy: ``(n, 2)`` tower plan coordinates.
        u_h_by_direction: Reference speed per direction, or one speed for all.
        n_segments / min_cycles: As in :func:`spectral_peak`.
        sigma: Deprecated and ignored.

    Returns:
        A ``pandas.DataFrame`` sorted by direction.
    """
    import pandas as pd

    _warn_sigma(sigma, "strouhal_by_direction")

    rows = []
    for direction in sorted(load_by_direction):
        source = load_by_direction[direction]
        u_h = u_h_by_direction if np.isscalar(u_h_by_direction) else u_h_by_direction[direction]
        width = across_wind_width(footprint_xy, direction)
        observed = _peak_frequency(
            across_wind_series(source, direction),
            float(source.time.timestep_size),
            band=None,
            n_segments=n_segments,
            min_cycles=min_cycles,
        )
        st = implied_strouhal(observed, u_h, width)
        rows.append(
            {
                "direction_deg": float(direction),
                "across_wind_width_m": round(width, 2),
                "u_h_ms": round(float(u_h), 3),
                "expected_hz": round(vortex_shedding_frequency(u_h, width, strouhal=strouhal), 4),
                "observed_hz": round(observed, 4),
                "implied_strouhal": round(st, 4),
                "passed": bool(physical_range[0] <= st <= physical_range[1]),
            }
        )
    return pd.DataFrame(rows)


def plot_strouhal_by_direction(
    table,
    *,
    strouhal: float = STROUHAL_RECTANGULAR,
    physical_range: tuple[float, float] = STROUHAL_PHYSICAL_RANGE,
    language: str = "en",
    ax=None,
):
    """Implied Strouhal number against wind direction, with the physical band.

    The figure is the case-quality read: points inside the shaded band and near
    the nominal line mean the load record is dimensionally consistent for that
    direction. A whole curve displaced from the band is the signature of a
    mis-scaled time axis.
    """
    import matplotlib.pyplot as plt
    from matplotlib.ticker import FuncFormatter

    if ax is None:
        _fig, ax = plt.subplots(figsize=(9, 4.5), layout="constrained")
    fig = ax.figure

    txt = {
        "en": (
            "wind direction",
            "implied Strouhal number",
            "physical range",
            f"nominal St = {strouhal:g}",
            "measured",
            "outside the range",
        ),
        "pt-br": (
            "direcao do vento",
            "numero de Strouhal implicito",
            "faixa fisica",
            f"St nominal = {strouhal:g}",
            "medido",
            "fora da faixa",
        ),
    }[language]

    d = table["direction_deg"].to_numpy(float)
    st = table["implied_strouhal"].to_numpy(float)
    ok = table["passed"].to_numpy(bool)

    ax.axhspan(physical_range[0], physical_range[1], color="#2F993A", alpha=0.12, label=txt[2])
    ax.axhline(strouhal, color="#2F993A", linestyle="--", linewidth=1.5, label=txt[3])
    ax.plot(d, st, "-", color="#E69F00", alpha=0.8, zorder=2)
    ax.plot(d[ok], st[ok], "o", color="#E69F00", label=txt[4], zorder=3)
    if (~ok).any():
        ax.plot(d[~ok], st[~ok], "X", color="#A82D2D", markersize=10, label=txt[5], zorder=4)

    ax.set_xlabel(txt[0])
    ax.set_ylabel(txt[1])
    ax.set_xticks(np.arange(0, d.max() + 1, 45))
    ax.xaxis.set_major_formatter(FuncFormatter(lambda v, _: rf"${v:g}^\circ$"))
    ax.set_ylim(0, max(float(st.max()) * 1.15, physical_range[1] * 1.15))
    ax.grid(alpha=0.3)
    ax.legend(loc="upper right", ncols=2, framealpha=0.9)
    return fig, ax
