"""Known-answer + broadcasting tests for the SDOF RK45 modal solver.

These tests validate the ported physics against closed-form analytical
solutions of the single-degree-of-freedom oscillator, with no dependency
on the legacy ``cfdmod.hfpi`` module.
"""

from __future__ import annotations

import numpy as np

from cfdmod.adapters.memory import MemoryFieldStore
from cfdmod.core import ElementMeta, ModesDataSource, TimeAxis
from cfdmod.core.recipes import sdof_rk45_solver
from cfdmod.core.recipes.dynamic import _solve_sdof_rk45


def _modes_source(q: np.ndarray, dt: float) -> ModesDataSource:
    """Wrap a (n_modes, n_t) generalized-load array in a ModesDataSource."""
    n_modes, n_t = q.shape
    return ModesDataSource(
        time=TimeAxis(initial_time=0.0, timestep_size=dt, n_timesteps=n_t),
        topology=None,
        elements=ElementMeta(),
        fields=MemoryFieldStore({"q": q}),
    )


def test_constant_forcing_reaches_static_deflection():
    """Constant load F0 -> steady state x_ss = F0 / wp^2 (mass-normalized)."""
    dt, wp, xi = 0.01, 2.0 * np.pi * 1.0, 0.02
    f0 = 3.0
    n_t = 4000
    q = np.full(n_t, f0)

    x = _solve_sdof_rk45(q, dt=dt, wp=wp, xi=xi)

    x_ss = f0 / wp**2
    # Compare a tail window (transient has decayed).
    np.testing.assert_allclose(x[-200:].mean(), x_ss, rtol=1e-3)
    assert np.isfinite(x).all()


def test_sinusoidal_forcing_matches_frequency_response_amplitude():
    """Sinusoidal load -> steady-state amplitude from the SDOF transfer function.

    |X| = F0 / sqrt((wp^2 - w^2)^2 + (2*xi*wp*w)^2)
    """
    dt, wp, xi = 0.005, 2.0 * np.pi * 2.0, 0.03
    f0 = 1.5
    w = 2.0 * np.pi * 1.3  # drive off-resonance
    n_t = 12000
    t = np.arange(n_t) * dt
    q = f0 * np.sin(w * t)

    x = _solve_sdof_rk45(q, dt=dt, wp=wp, xi=xi)

    amp_expected = f0 / np.sqrt((wp**2 - w**2) ** 2 + (2 * xi * wp * w) ** 2)
    # Steady-state amplitude from the tail (peak-to-peak / 2).
    tail = x[-4000:]
    amp_observed = 0.5 * (tail.max() - tail.min())
    np.testing.assert_allclose(amp_observed, amp_expected, rtol=2e-2)


def test_resonance_amplification():
    """At resonance (w == wp) the response amplitude approaches F0 / (2*xi*wp^2)."""
    dt, wp, xi = 0.005, 2.0 * np.pi * 2.0, 0.05
    f0 = 1.0
    w = wp
    n_t = 20000
    t = np.arange(n_t) * dt
    q = f0 * np.sin(w * t)

    x = _solve_sdof_rk45(q, dt=dt, wp=wp, xi=xi)

    amp_expected = f0 / (2 * xi * wp**2)
    tail = x[-6000:]
    amp_observed = 0.5 * (tail.max() - tail.min())
    np.testing.assert_allclose(amp_observed, amp_expected, rtol=5e-2)


def test_solver_factory_multimode_broadcasting():
    """Scalar damping broadcasts; per-mode frequencies map to the right rows."""
    dt = 0.01
    n_t = 4000
    f0 = np.array([2.0, 5.0])[:, None]
    q = np.tile(f0, (1, n_t))  # constant forcing per mode
    wps = np.array([2 * np.pi * 1.0, 2 * np.pi * 3.0])

    modes = _modes_source(q, dt)
    solver = sdof_rk45_solver(natural_frequencies=wps, damping_ratio=0.02)
    solved = solver(modes)

    x = np.asarray(solved.fields.read("q"))
    assert x.shape == (2, n_t)
    x_ss = f0[:, 0] / wps**2
    np.testing.assert_allclose(x[:, -200:].mean(axis=1), x_ss, rtol=1e-3)


def test_solver_factory_rejects_frequency_count_mismatch():
    import pytest

    modes = _modes_source(np.ones((3, 100)), dt=0.01)
    solver = sdof_rk45_solver(natural_frequencies=[1.0, 2.0], damping_ratio=0.02)
    with pytest.raises(ValueError, match="expected n_modes=3"):
        solver(modes)
