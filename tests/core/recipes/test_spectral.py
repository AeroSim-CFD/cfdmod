"""Spectral modal response: analytical answers and agreement with the SDOF time solve.

Pinned three ways:

* the integration itself, on a white load spectrum built by hand (closed form
  ``sigma^2 = S0 / (8 xi w^3)`` for one mode, one-sided S0 per Hz, no Welch, no randomness);
* the whole chain (Welch cross-spectra -> admittance -> sigma) against the exact
  SDOF time solve of the same load record, for displacement and acceleration;
* two modes 1% apart driven by correlated loads: the cross terms are what makes the
  spectral sigma agree with the time solve, and summing the modes independently
  (SRSS) is visibly off.
"""

from __future__ import annotations

import numpy as np
import pytest
from scipy import signal

from cfdmod.core.recipes.dynamic import _solve_sdof_exact
from cfdmod.core.recipes.spectral import (
    modal_admittance,
    modal_load_csd,
    spectral_response,
)


def _low_pass_noise(n_modes: int, n_t: int, dt: float, f_cut: float, seed: int) -> np.ndarray:
    """Band-limited Gaussian loads with a smooth spectrum (like a turbulent buffeting load)."""
    rng = np.random.default_rng(seed)
    b, a = signal.butter(2, f_cut, fs=1.0 / dt)
    return signal.lfilter(b, a, rng.standard_normal((n_modes, n_t)), axis=1)


@pytest.mark.unit
def test_white_spectrum_single_mode_closed_form():
    s0, fn, xi = 2.0, 0.25, 0.015
    wp = 2 * np.pi * fn
    freq = np.linspace(0.0, 5.0, 2001)
    csd = np.full((freq.size, 1, 1), s0, dtype=complex)
    r = spectral_response(freq, csd, np.zeros(1), np.ones((1, 1)), [wp], xi)
    # int_-inf^inf |H|^2 dw = pi / (2 xi w^3) -> int_0^inf |H|^2 df = 1 / (8 xi w^3);
    # the tail above 5 Hz is negligible (~1e-6)
    assert r.sigma[0] == pytest.approx(np.sqrt(s0 / (8 * xi * wp**3)), rel=2e-3)
    assert r.nu[0] == pytest.approx(fn, rel=0.02)


@pytest.mark.unit
def test_mean_is_static_response_and_load_scale():
    freq = np.linspace(0.0, 2.0, 101)
    csd = np.zeros((freq.size, 2, 2), dtype=complex)
    wp = np.array([1.5, 3.0])
    q_mean = np.array([4.0, -9.0])
    c = np.array([[1.0, 2.0]])
    r = spectral_response(freq, csd, q_mean, c, wp, 0.02, load_scale=0.5)
    assert r.mean[0] == pytest.approx(0.5 * (4.0 / 1.5**2 - 2 * 9.0 / 3.0**2))
    assert r.sigma[0] == 0.0
    acc = spectral_response(freq, csd, q_mean, c, wp, 0.02, derivative=2)
    assert acc.mean[0] == 0.0


@pytest.mark.unit
def test_admittance_static_and_resonant_values():
    wp, xi = 2.0, 0.05
    h = modal_admittance(np.array([0.0, wp / (2 * np.pi)]), [wp], xi)[:, 0]
    assert h[0] == pytest.approx(1 / wp**2)
    assert abs(h[1]) == pytest.approx(1 / (2 * xi * wp**2))


@pytest.mark.unit
@pytest.mark.parametrize("derivative", [0, 2])
def test_matches_time_solve_single_mode(derivative):
    dt, n_t = 0.1, 2**18
    fn, xi = 0.22, 0.015
    wp = 2 * np.pi * fn
    q = _low_pass_noise(1, n_t, dt, f_cut=0.5, seed=1) + 3.0
    x = _solve_sdof_exact(q, dt, np.array([wp]), np.array([xi]))[0]
    resp = x if derivative == 0 else np.gradient(np.gradient(x, dt), dt)
    freq, csd, q_mean = modal_load_csd(q, dt, n_segments=64)
    r = spectral_response(freq, csd, q_mean, np.ones((1, 1)), [wp], xi, derivative=derivative)
    if derivative == 0:
        assert r.mean[0] == pytest.approx(x.mean(), rel=1e-3)
    assert r.sigma[0] == pytest.approx(resp.std(), rel=0.05)


@pytest.mark.unit
def test_close_modes_need_the_cross_terms():
    dt, n_t = 0.1, 2**18
    wp = 2 * np.pi * np.array([0.221, 0.223])
    xi = np.array([0.015, 0.015])
    common = _low_pass_noise(1, n_t, dt, f_cut=0.5, seed=2)
    q = np.vstack([common[0], 0.8 * common[0]]) + 0.2 * _low_pass_noise(2, n_t, dt, 0.5, seed=3)
    x = _solve_sdof_exact(q, dt, wp, xi)
    c = np.array([[1.0, 1.0]])  # a floor where both modes move the same way
    freq, csd, q_mean = modal_load_csd(q, dt, n_segments=64)
    r = spectral_response(freq, csd, q_mean, c, wp, xi)
    time_sigma = (c @ x)[0].std()
    assert r.sigma[0] == pytest.approx(time_sigma, rel=0.05)
    srss = np.sqrt(
        sum(
            spectral_response(freq, csd, q_mean, np.eye(2)[[m]], wp, xi).sigma[0] ** 2
            for m in range(2)
        )
    )
    assert abs(srss / time_sigma - 1) > 0.2


@pytest.mark.unit
def test_csd_is_hermitian_and_matches_welch_auto_spectrum():
    dt = 0.05
    q = _low_pass_noise(3, 4096, dt, 1.0, seed=4)
    freq, csd, _ = modal_load_csd(q, dt, n_segments=8)
    assert np.allclose(csd, np.conj(np.swapaxes(csd, 1, 2)))
    _, p11 = signal.welch(q[1], fs=1 / dt, nperseg=2 * 4096 // 9, noverlap=4096 // 9)
    assert np.allclose(csd[:, 1, 1].real, p11)


@pytest.mark.unit
def test_shape_errors():
    freq = np.linspace(0, 1, 11)
    with pytest.raises(ValueError):
        spectral_response(freq, np.zeros((11, 2, 2)), np.zeros(2), np.ones((1, 3)), [1, 2], 0.01)
    with pytest.raises(ValueError):
        spectral_response(
            freq, np.zeros((11, 1, 1)), np.zeros(1), np.ones((1, 1)), [1], 0.01, derivative=3
        )
    with pytest.raises(ValueError):
        modal_load_csd(np.zeros(10), 0.1)
