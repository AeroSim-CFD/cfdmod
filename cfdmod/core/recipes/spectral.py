"""Spectral (frequency-domain) modal response -- the counterpart of the SDOF time solve.

For a linear modal model with mass-normalised modes, ``q'' + 2 xi w q' + w^2 q = Q(t)``, every
response that is a linear combination of the modal coordinates, ``r(t) = sum_m c_m q_m(t)``, has

    mean    r_mean = sum_m c_m Q_mean_m / w_m^2
    PSD     S_r(f) = a(f)^H P(f) a(f),   a_m(f) = c_m H_m(f)
            H_m(f) = 1 / (w_m^2 - w^2 + 2 i xi_m w_m w),   w = 2 pi f

with ``P(f)`` the one-sided cross-spectral matrix of the generalized loads. All mode pairs enter:
modes with close frequencies respond to correlated loads, and summing the modes independently
(SRSS) misses that. A time derivative of order ``k`` multiplies ``S_r`` by ``w^(2k)`` (``k = 2``:
acceleration). From the spectrum: ``sigma^2 = int S_r df`` and the mean up-crossing rate
``nu = sqrt(int f^2 S_r df / sigma^2)`` (the Davenport peak factor frequency).

The load cross-spectra are Welch averages: smooth, at the coarse resolution a finite record
allows. The mechanical admittance ``|H|^2`` is analytic and, at resonance, much narrower (half-power
bandwidth ``2 xi f_m``), so the integral runs on a grid refined around every natural frequency with
``P`` interpolated onto it: the resonant area is exact for a load spectrum smooth across the
bandwidth, instead of riding on the noise of a single record.

Why it exists next to the time solve: the loads and mode shapes do not change when only the
frequencies, damping or mass of the model change, so ``P`` is computed once per load case and every
structural variant is an integral -- the sensitivity sweeps run in milliseconds per case and give
the trend of the response, not the sampling scatter of one record.
"""

from __future__ import annotations

__all__ = [
    "SpectralResponse",
    "modal_load_csd",
    "modal_admittance",
    "spectral_response",
]

from dataclasses import dataclass

import numpy as np
from scipy import signal


@dataclass(frozen=True)
class SpectralResponse:
    """Statistics of ``n_resp`` responses (each array has shape ``(n_resp,)``).

    Attributes:
        mean: static response to the mean loads (zero for ``derivative > 0``).
        sigma: standard deviation of the fluctuating response.
        nu: mean up-crossing rate [Hz], ``sqrt(m2 / m0)`` of the response spectrum.
    """

    mean: np.ndarray
    sigma: np.ndarray
    nu: np.ndarray


def modal_load_csd(
    q: np.ndarray, dt: float, *, n_segments: int = 8
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """One-sided cross-spectral matrix of the generalized loads (Welch, Hann, 50% overlap).

    Args:
        q: ``(n_modes, n_t)`` generalized loads (e.g. the ``q`` field of
            :func:`cfdmod.core.ops.data_source_create.generalized_building_load`).
        dt: time step [s].
        n_segments: Welch segments; more = smoother spectrum, coarser resolution
            (segment length ``2 n_t / (n_segments + 1)`` samples).

    Returns:
        ``(freq, P, q_mean)``: ``freq`` ``(n_f,)`` [Hz]; ``P`` ``(n_f, n_modes, n_modes)``
        complex, Hermitian, ``P[f, m, n] = E[conj(Q_m) Q_n]`` (one-sided, per Hz, mean
        removed); ``q_mean`` ``(n_modes,)``.
    """
    q = np.asarray(q, dtype=np.float64)
    if q.ndim != 2:
        raise ValueError(f"q must be 2-D (n_modes, n_t); got {q.shape}")
    if not np.isfinite(dt) or dt <= 0:
        raise ValueError(f"dt must be positive and finite; got {dt}")
    n_t = q.shape[1]
    nperseg = int(2 * n_t // (n_segments + 1))
    if nperseg < 16:
        raise ValueError(f"record too short for {n_segments} segments: {n_t} samples")
    freq, p = signal.csd(
        q[:, None, :],
        q[None, :, :],
        fs=1.0 / dt,
        window="hann",
        nperseg=nperseg,
        noverlap=nperseg // 2,
        detrend="constant",
        scaling="density",
        axis=-1,
    )
    return freq, np.moveaxis(p, -1, 0), q.mean(axis=1)


def modal_admittance(freq: np.ndarray, natural_frequencies, damping_ratio) -> np.ndarray:
    """``H_m(f) = 1 / (w_m^2 - w^2 + 2 i xi_m w_m w)``, shape ``(n_f, n_modes)``.

    ``natural_frequencies`` in rad/s (as the structural model carries them), unit generalized mass.
    """
    wp = np.atleast_1d(np.asarray(natural_frequencies, dtype=np.float64))
    xi = np.broadcast_to(np.atleast_1d(np.asarray(damping_ratio, dtype=np.float64)), wp.shape)
    w = 2.0 * np.pi * np.asarray(freq, dtype=np.float64)[:, None]
    return 1.0 / (wp**2 - w**2 + 2j * xi * wp * w)


def _refined_grid(
    freq: np.ndarray, wp: np.ndarray, xi: np.ndarray, half_width: float, n_band: int
):
    """The Welch grid plus ``n_band`` points across ``f_m (1 +- half_width xi_m)`` of every mode."""
    f_max = float(freq[-1])
    bands = [freq]
    for fm, x in zip(wp / (2.0 * np.pi), xi):
        if fm < f_max:
            bands.append(fm + fm * max(x, 1e-4) * np.linspace(-half_width, half_width, n_band))
    grid = np.unique(np.concatenate(bands))
    return grid[(grid >= freq[0]) & (grid <= f_max)]


def _interp_rows(grid: np.ndarray, freq: np.ndarray, p: np.ndarray) -> np.ndarray:
    """Linear interpolation of ``p`` (``(n_f, ...)``) from ``freq`` onto ``grid`` along axis 0."""
    i = np.clip(np.searchsorted(freq, grid, side="right") - 1, 0, freq.size - 2)
    t = (grid - freq[i]) / (freq[i + 1] - freq[i])
    t = t.reshape((-1,) + (1,) * (p.ndim - 1))
    return p[i] * (1.0 - t) + p[i + 1] * t


def spectral_response(
    freq: np.ndarray,
    csd: np.ndarray,
    q_mean: np.ndarray,
    coeffs: np.ndarray,
    natural_frequencies,
    damping_ratio,
    *,
    derivative: int = 0,
    load_scale: float = 1.0,
    half_width: float = 50.0,
    n_band: int = 401,
) -> SpectralResponse:
    """Mean, sigma and up-crossing rate of ``r = coeffs @ q`` (and its time derivatives).

    Args:
        freq, csd, q_mean: output of :func:`modal_load_csd`.
        coeffs: ``(n_resp, n_modes)`` -- each row a response as a combination of the
            modal coordinates (e.g. ``phi[floor, :, 0]`` for a floor x displacement;
            ``mass * phi * w_m^2`` for a static-equivalent load).
        natural_frequencies: rad/s, ``(n_modes,)``.
        damping_ratio: scalar or ``(n_modes,)``.
        derivative: 0 displacement-like, 1 velocity, 2 acceleration.
        load_scale: multiplies the generalized loads (``csd`` by its square): the loads of
            mode shapes scaled by a constant, without recomputing the spectra.
        half_width, n_band: refined band ``f_m (1 +- half_width xi_m)`` with ``n_band``
            points per mode (defaults resolve the resonant peak to well under 1%).

    Modes above the last frequency of ``freq`` (the Nyquist of the record) get no load: the
    record cannot excite them, the same limit as the time solve.
    """
    coeffs = np.atleast_2d(np.asarray(coeffs, dtype=np.float64))
    wp = np.atleast_1d(np.asarray(natural_frequencies, dtype=np.float64))
    xi = np.broadcast_to(np.atleast_1d(np.asarray(damping_ratio, dtype=np.float64)), wp.shape)
    freq = np.asarray(freq, dtype=np.float64)
    csd = np.asarray(csd)
    n_modes = wp.size
    if coeffs.shape[1] != n_modes or csd.shape[1:] != (n_modes, n_modes):
        raise ValueError(
            f"coeffs {coeffs.shape} / csd {csd.shape} do not match {n_modes} natural frequencies"
        )
    if derivative not in (0, 1, 2):
        raise ValueError(f"derivative must be 0, 1 or 2; got {derivative}")

    grid = _refined_grid(freq, wp, xi, half_width, n_band)
    p = _interp_rows(grid, freq, csd) * load_scale**2
    a = coeffs[:, None, :] * modal_admittance(grid, wp, xi)[None]  # (n_resp, n_g, n_modes)
    s = np.einsum("rgm,gmn,rgn->rg", a.conj(), p, a).real
    if derivative:
        s = s * (2.0 * np.pi * grid) ** (2 * derivative)
    s = np.clip(s, 0.0, None)  # round-off of a Hermitian form
    m0 = np.trapezoid(s, grid, axis=1)
    m2 = np.trapezoid(s * grid**2, grid, axis=1)
    nu = np.sqrt(np.divide(m2, m0, out=np.zeros_like(m0), where=m0 > 0))
    mean = coeffs @ (np.asarray(q_mean, dtype=np.float64) * load_scale / wp**2)
    if derivative:
        mean = np.zeros_like(mean)
    return SpectralResponse(mean=mean, sigma=np.sqrt(m0), nu=nu)
