"""Peak-estimation methods for building response time series.

The engineer-facing deliverables reduce a fluctuating response (acceleration,
floor force, displacement) to a single design peak. Three methods are used in
the field, selectable per deliverable:

- ``"max"`` -- the observed extreme (optionally of the absolute value).
- ``"peak-factor"`` -- Davenport: ``mean + g * std`` with the gust peak factor
  ``g`` from the response frequency and averaging duration.
- ``"gumbel"`` -- fit a Gumbel to block maxima and read the design fractile off
  it (more stable than the raw max for short records).
"""

from __future__ import annotations

from typing import Literal

import numpy as np

PeakMethod = Literal["max", "peak-factor", "gumbel"]


def gust_peak_factor(f0: float, duration: float = 600.0, *, full: bool = True) -> float:
    """Davenport gust peak factor ``g`` for a narrow-band process.

    ``g = sqrt(2 ln(nu T)) + 0.5772 / sqrt(2 ln(nu T))`` with ``nu = f0`` the
    mean up-crossing rate (Hz) and ``T = duration`` (s). With ``full=False``
    only the leading ``sqrt(2 ln(nu T))`` term is returned (the form used in the
    quick-look notebooks).
    """
    nu_t = f0 * duration
    if nu_t <= 1.0:
        raise ValueError(f"f0 * duration must exceed 1 (got {nu_t})")
    base = np.sqrt(2.0 * np.log(nu_t))
    return float(base + 0.5772 / base if full else base)


def _gumbel_fit(block_maxima: np.ndarray) -> tuple[float, float]:
    """Method-of-moments Gumbel (loc, scale) from block maxima."""
    m = float(np.mean(block_maxima))
    s = float(np.std(block_maxima, ddof=1)) if block_maxima.size > 1 else 0.0
    scale = s * np.sqrt(6.0) / np.pi
    loc = m - 0.5772 * scale
    return loc, scale


def peak_value(
    series: np.ndarray,
    method: PeakMethod = "peak-factor",
    *,
    f0: float | None = None,
    duration: float = 600.0,
    absolute: bool = True,
    n_blocks: int = 10,
    non_exceedance: float = 0.78,
) -> float:
    """Reduce a response time series to a single design peak.

    Args:
        method: ``"max"``, ``"peak-factor"`` (needs ``f0``), or ``"gumbel"``.
        f0: response frequency (Hz), required for ``"peak-factor"``.
        duration: full-scale averaging window (s) for the gust factor.
        absolute: if True, ``"max"`` uses ``max(|series|)`` and ``"peak-factor"``
            builds the peak off ``|mean| + g*std``.
        n_blocks: number of blocks for ``"gumbel"`` block maxima.
        non_exceedance: design fractile ``p`` for ``"gumbel"``
            (``x_p = loc - scale ln(-ln p)``).
    """
    x = np.asarray(series, dtype=np.float64)
    x = x[np.isfinite(x)]
    if x.size == 0:
        return float("nan")

    if method == "max":
        return float(np.max(np.abs(x)) if absolute else np.max(x))

    if method == "peak-factor":
        if f0 is None:
            raise ValueError("peak-factor method requires f0")
        g = gust_peak_factor(f0, duration)
        mean = abs(float(np.mean(x))) if absolute else float(np.mean(x))
        return float(mean + g * float(np.std(x)))

    if method == "gumbel":
        vals = np.abs(x) if absolute else x
        n_blocks = max(1, min(n_blocks, vals.size))
        blocks = np.array_split(vals, n_blocks)
        block_maxima = np.array([float(np.max(b)) for b in blocks if b.size])
        loc, scale = _gumbel_fit(block_maxima)
        if scale == 0.0:
            return float(loc)
        return float(loc - scale * np.log(-np.log(non_exceedance)))

    raise ValueError(f"unknown peak method {method!r}")
