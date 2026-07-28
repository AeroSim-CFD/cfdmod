"""Exact (Nigam-Jennings) SDOF modal solver: known answers, agreement, speed.

Regression cover for two defects found on a real high-rise case:

* the adaptive RK45 path ran at SciPy's default tolerances and carried ~3%
  error on a lightly-damped modal response -- enough to move a design peak;
* it was also ~3 orders of magnitude slower than the closed-form recurrence,
  which made a full directional fan-out an overnight job.

The closed-form solver is exact for the piecewise-linear forcing the RK45 path
already assumed, so these are the same physics with the integration error and
the cost removed.
"""

from __future__ import annotations

import time

import numpy as np
import pytest

from cfdmod.adapters.memory import MemoryFieldStore
from cfdmod.core import ElementMeta, ModesDataSource, TimeAxis
from cfdmod.core.recipes import sdof_exact_solver, sdof_solver
from cfdmod.core.recipes.dynamic import _solve_sdof_exact, _solve_sdof_rk45


def _modes_source(q: np.ndarray, dt: float) -> ModesDataSource:
    n_modes, n_t = q.shape
    return ModesDataSource(
        time=TimeAxis(initial_time=0.0, timestep_size=dt, n_timesteps=n_t),
        topology=None,
        elements=ElementMeta(),
        fields=MemoryFieldStore({"q": q}),
    )


def _broadband_load(n_t: int, n_modes: int = 1, seed: int = 0) -> np.ndarray:
    """Random-walk load: broadband, so every mode is genuinely excited."""
    rng = np.random.default_rng(seed)
    return rng.standard_normal((n_modes, n_t)).cumsum(axis=1) * 0.01


@pytest.mark.unit
def test_constant_forcing_reaches_static_deflection():
    dt, wp, xi, f0 = 0.01, 2.0 * np.pi, 0.02, 3.0
    q = np.full((1, 4000), f0)
    x = _solve_sdof_exact(q, dt, np.array([wp]), np.array([xi]))[0]
    np.testing.assert_allclose(x[-200:].mean(), f0 / wp**2, rtol=1e-6)


@pytest.mark.unit
def test_sinusoidal_forcing_matches_frequency_response():
    """|X| = F0 / sqrt((wp^2 - w^2)^2 + (2 xi wp w)^2), the SDOF transfer function."""
    dt, wp, xi, f0 = 0.005, 2.0 * np.pi * 2.0, 0.03, 1.5
    w = 2.0 * np.pi * 1.3
    n_t = 8000
    t = np.arange(n_t) * dt
    q = (f0 * np.sin(w * t))[None, :]

    x = _solve_sdof_exact(q, dt, np.array([wp]), np.array([xi]))[0]

    expected = f0 / np.sqrt((wp**2 - w**2) ** 2 + (2 * xi * wp * w) ** 2)
    tail = x[n_t // 2 :]
    np.testing.assert_allclose(np.abs(tail).max(), expected, rtol=2e-3)


@pytest.mark.unit
def test_resonant_amplification_matches_1_over_2xi():
    """At w = wp the steady amplitude is F0 / (2 xi wp^2) -- the resonance check.

    This is the quantity a wrong time axis corrupts, so it gets its own test.
    """
    xi, wp, f0 = 0.02, 2.0 * np.pi * 0.5, 1.0
    dt = 0.01
    n_t = 60000  # long enough for the lightly-damped transient to settle
    t = np.arange(n_t) * dt
    q = (f0 * np.sin(wp * t))[None, :]

    x = _solve_sdof_exact(q, dt, np.array([wp]), np.array([xi]))[0]

    expected = f0 / (2 * xi * wp**2)
    np.testing.assert_allclose(np.abs(x[-20000:]).max(), expected, rtol=5e-3)


@pytest.mark.unit
@pytest.mark.parametrize("wp,xi", [(1.3437, 0.02), (5.5, 0.02), (0.5, 0.015)])
def test_exact_agrees_with_rk45(wp, xi):
    """The two solvers integrate the same ODE, so they must converge on it.

    The RK45 path is only this close because its tolerances were tightened; at
    SciPy's defaults it drifts by ~3%, which is the regression this guards.
    """
    dt, n_t = 0.05, 4000
    q = _broadband_load(n_t)[0]

    exact = _solve_sdof_exact(q[None, :], dt, np.array([wp]), np.array([xi]))[0]
    rk45 = _solve_sdof_rk45(q, dt=dt, wp=wp, xi=xi)

    assert np.sqrt(((exact - rk45) ** 2).mean()) / rk45.std() < 5e-3


@pytest.mark.unit
def test_exact_solver_is_vectorised_over_modes():
    """Solving 4 modes together must equal solving each on its own."""
    dt, n_t = 0.02, 2000
    wps = np.array([1.3, 2.7, 5.5, 9.1])
    xis = np.full(4, 0.02)
    q = _broadband_load(n_t, n_modes=4, seed=3)

    together = _solve_sdof_exact(q, dt, wps, xis)
    for m in range(4):
        alone = _solve_sdof_exact(q[m : m + 1], dt, wps[m : m + 1], xis[m : m + 1])
        np.testing.assert_allclose(together[m], alone[0], rtol=1e-12, atol=1e-14)


@pytest.mark.unit
def test_solver_wrapper_replaces_q_field():
    dt, n_t = 0.02, 500
    q = _broadband_load(n_t, n_modes=2, seed=5)
    solver = sdof_exact_solver(natural_frequencies=[1.3, 4.4], damping_ratio=0.02)
    out = solver(_modes_source(q, dt))
    assert np.asarray(out.fields.read("q")).shape == q.shape
    assert not np.allclose(np.asarray(out.fields.read("q")), q)


@pytest.mark.unit
def test_sdof_solver_selects_method():
    dt, n_t = 0.02, 300
    q = _broadband_load(n_t, n_modes=1, seed=7)
    src = _modes_source(q, dt)
    a = sdof_solver(natural_frequencies=[2.0], damping_ratio=0.02, method="exact")(src)
    b = sdof_solver(natural_frequencies=[2.0], damping_ratio=0.02, method="rk45")(src)
    assert np.allclose(
        np.asarray(a.fields.read("q")), np.asarray(b.fields.read("q")), rtol=0, atol=1e-3
    )
    with pytest.raises(ValueError, match="unknown sdof method"):
        sdof_solver(natural_frequencies=[2.0], damping_ratio=0.02, method="nope")


@pytest.mark.unit
def test_exact_solver_rejects_supercritical_damping():
    q = _broadband_load(100)
    solver = sdof_exact_solver(natural_frequencies=[2.0], damping_ratio=1.5)
    with pytest.raises(ValueError, match="sub-critical damping"):
        solver(_modes_source(q, 0.02))


@pytest.mark.perf
def test_exact_solver_is_orders_of_magnitude_faster_than_rk45():
    """A 10-mode, 4000-step solve is the per-direction unit of a real fan-out."""
    dt, n_t = 0.19, 4000
    wps = np.array([1.34, 1.65, 1.83, 4.42, 5.18, 5.43, 5.75, 8.35, 8.58, 9.19])
    xis = np.full(10, 0.02)
    q = _broadband_load(n_t, n_modes=10, seed=11)

    t0 = time.perf_counter()
    _solve_sdof_exact(q, dt, wps, xis)
    exact_s = time.perf_counter() - t0

    t0 = time.perf_counter()
    for m in range(2):  # two modes is enough to establish the ratio
        _solve_sdof_rk45(q[m], dt=dt, wp=float(wps[m]), xi=0.02)
    rk45_s = (time.perf_counter() - t0) / 2 * 10

    assert exact_s < 0.5, f"exact solver took {exact_s:.3f}s for 10 modes x {n_t} steps"
    assert rk45_s / exact_s > 100, f"expected >100x speedup, got {rk45_s / exact_s:.0f}x"
