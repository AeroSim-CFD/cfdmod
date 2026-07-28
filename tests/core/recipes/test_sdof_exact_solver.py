"""SDOF modal solver: known answers, an independent integrator, and speed.

The solver integrates the mass-normalized modal equation

    x'' + 2 xi wp x' + wp^2 x = Q(t)

in closed form per step (Nigam-Jennings), which is exact when ``Q`` varies
linearly across a step. Correctness is pinned two ways: against the analytical
SDOF responses (static deflection, the transfer function, resonance), and
against an independent adaptive integrator defined here in the tests -- so the
cross-check does not require shipping a second solver in the library.

Regression cover for two defects found on a real high-rise case:

* the library previously integrated this ODE with ``solve_ivp`` at SciPy's
  default tolerances, which carries ~3% error on a lightly-damped modal
  response -- enough to move a design peak;
* that path was also ~3 orders of magnitude slower, which made a full
  directional fan-out an overnight job.
"""

from __future__ import annotations

import time

import numpy as np
import pytest
from scipy import integrate
from scipy.interpolate import interp1d

from cfdmod.adapters.memory import MemoryFieldStore
from cfdmod.core import ElementMeta, ModesDataSource, TimeAxis
from cfdmod.core.recipes import sdof_exact_solver
from cfdmod.core.recipes.dynamic import _sdof_seed, _solve_sdof_exact


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


def _reference_rk45(q: np.ndarray, dt: float, wp: float, xi: float) -> np.ndarray:
    """Independent adaptive integration of the same ODE, at tight tolerances.

    Deliberately a different numerical method from the one under test, and
    deliberately local to the tests: the library ships one solver, and this is
    how we check it rather than a second production code path. The forcing is
    interpolated linearly, which is the assumption the closed form makes exact.
    """
    t = np.arange(len(q)) * dt
    f = interp1d(t, q, kind="linear", fill_value="extrapolate")
    x0, v0 = _sdof_seed(q, dt, wp, xi)
    sol = integrate.solve_ivp(
        lambda tt, y: [y[1], f(tt) - 2 * xi * wp * y[1] - wp**2 * y[0]],
        (t[0], t[-1]),
        [x0, v0],
        t_eval=t,
        method="RK45",
        rtol=1e-9,
        atol=1e-12,
    )
    return sol.y[0]


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
def test_matches_an_independent_integrator(wp, xi):
    """Closed form vs a converged adaptive integration of the same ODE."""
    dt, n_t = 0.05, 4000
    q = _broadband_load(n_t)[0]

    exact = _solve_sdof_exact(q[None, :], dt, np.array([wp]), np.array([xi]))[0]
    reference = _reference_rk45(q, dt, wp, xi)

    assert np.sqrt(((exact - reference) ** 2).mean()) / reference.std() < 1e-5


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
def test_solver_factory_broadcasts_scalar_damping_over_modes():
    """Scalar damping broadcasts; per-mode frequencies map to the right rows."""
    dt, n_t = 0.01, 4000
    f0 = np.array([2.0, 5.0])[:, None]
    q = np.tile(f0, (1, n_t))  # constant forcing per mode
    wps = np.array([2 * np.pi * 1.0, 2 * np.pi * 3.0])

    solver = sdof_exact_solver(natural_frequencies=wps, damping_ratio=0.02)
    x = np.asarray(solver(_modes_source(q, dt)).fields.read("q"))

    assert x.shape == (2, n_t)
    np.testing.assert_allclose(x[:, -200:].mean(axis=1), f0[:, 0] / wps**2, rtol=1e-6)


@pytest.mark.unit
def test_solver_wrapper_replaces_q_field():
    dt, n_t = 0.02, 500
    q = _broadband_load(n_t, n_modes=2, seed=5)
    solver = sdof_exact_solver(natural_frequencies=[1.3, 4.4], damping_ratio=0.02)
    out = solver(_modes_source(q, dt))
    assert np.asarray(out.fields.read("q")).shape == q.shape
    assert not np.allclose(np.asarray(out.fields.read("q")), q)


@pytest.mark.unit
def test_solver_rejects_frequency_count_mismatch():
    solver = sdof_exact_solver(natural_frequencies=[1.0, 2.0], damping_ratio=0.02)
    with pytest.raises(ValueError, match="expected n_modes=3"):
        solver(_modes_source(np.ones((3, 100)), dt=0.01))


@pytest.mark.unit
def test_exact_solver_rejects_supercritical_damping():
    q = _broadband_load(100)
    solver = sdof_exact_solver(natural_frequencies=[2.0], damping_ratio=1.5)
    with pytest.raises(ValueError, match="sub-critical damping"):
        solver(_modes_source(q, 0.02))


@pytest.mark.perf
def test_solve_is_fast_enough_for_a_directional_fanout():
    """A 10-mode, 4000-step solve is the per-direction unit of a real fan-out.

    16 directions x 2 recurrence periods x 2 damping ratios is 64 of these, so
    anything above ~0.1 s each turns the fan-out into an overnight job. The
    adaptive integrator this replaced took ~10 s per solve.
    """
    dt, n_t = 0.19, 4000
    wps = np.array([1.34, 1.65, 1.83, 4.42, 5.18, 5.43, 5.75, 8.35, 8.58, 9.19])
    xis = np.full(10, 0.02)
    q = _broadband_load(n_t, n_modes=10, seed=11)

    t0 = time.perf_counter()
    _solve_sdof_exact(q, dt, wps, xis)
    elapsed = time.perf_counter() - t0

    assert elapsed < 0.5, f"10 modes x {n_t} steps took {elapsed:.3f}s"
