"""Unit tests for the time-derivative field op.

Accuracy is checked against closed-form derivatives of an analytic
signal; the legacy boundary stencils are locked in against a hand-rolled
reference.
"""

from __future__ import annotations

import numpy as np
import pytest

from cfdmod.adapters.memory import MemoryFieldStore
from cfdmod.core import ElementMeta, SurfaceDataSource, TimeAxis, Topology
from cfdmod.core.ops.field.derivative import DerivativeParams, derivative


def _surface(values: np.ndarray, dt: float = 0.01) -> SurfaceDataSource:
    n_elements, n_timesteps = values.shape
    verts = np.tile(np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0]]), (n_elements, 1)).astype(
        np.float64
    )
    tris = (np.arange(n_elements * 3).reshape(n_elements, 3)).astype(np.int32)
    return SurfaceDataSource(
        time=TimeAxis(initial_time=0.0, timestep_size=dt, n_timesteps=n_timesteps),
        topology=Topology.triangles(tris, verts),
        elements=ElementMeta(),
        fields=MemoryFieldStore({"displacement": values.astype(np.float64)}),
    )


def test_first_derivative_default_out_name_is_velocity():
    data = np.arange(10.0).reshape(1, 10)
    ds = _surface(data, dt=1.0)
    out = derivative(ds, DerivativeParams(order=1))
    assert sorted(out.field_names) == ["displacement", "velocity"]


def test_second_derivative_default_out_name_is_acceleration():
    data = np.arange(10.0).reshape(1, 10) ** 2
    ds = _surface(data, dt=1.0)
    out = derivative(ds, DerivativeParams(order=2))
    assert sorted(out.field_names) == ["acceleration", "displacement"]


def test_first_derivative_linear_ramp_is_constant_slope():
    dt = 0.5
    slope = 3.0
    t = np.arange(20) * dt
    data = (slope * t).reshape(1, -1)
    ds = _surface(data, dt=dt)
    v = derivative(ds, DerivativeParams(order=1)).fields.read("velocity")
    np.testing.assert_allclose(v, slope, rtol=1e-12)


def test_first_derivative_matches_legacy_stencil():
    rng = np.random.default_rng(1)
    data = rng.normal(size=(3, 30))
    dt = 0.02
    ds = _surface(data, dt=dt)
    v = derivative(ds, DerivativeParams(order=1, out="v")).fields.read("v")

    expected = np.zeros_like(data)
    expected[:, 1:] = (data[:, 1:] - data[:, :-1]) / dt
    expected[:, 0] = (data[:, 1] - data[:, 0]) / dt
    np.testing.assert_allclose(v, expected, rtol=1e-12)


def test_second_derivative_of_quadratic_is_constant():
    dt = 0.1
    a = 2.0
    t = np.arange(50) * dt
    data = (a * t**2).reshape(1, -1)
    ds = _surface(data, dt=dt)
    acc = derivative(ds, DerivativeParams(order=2)).fields.read("acceleration")
    # d2/dt2 (a t^2) = 2a exactly for the central stencil.
    np.testing.assert_allclose(acc[:, 1:-1], 2 * a, rtol=1e-10)


def test_second_derivative_matches_legacy_stencil():
    rng = np.random.default_rng(2)
    data = rng.normal(size=(2, 25))
    dt = 0.05
    ds = _surface(data, dt=dt)
    acc = derivative(ds, DerivativeParams(order=2, out="acc")).fields.read("acc")

    expected = np.zeros_like(data)
    expected[:, 1:-1] = (data[:, 2:] - 2 * data[:, 1:-1] + data[:, :-2]) / dt**2
    expected[:, 0] = (data[:, 2] - 2 * data[:, 1] + data[:, 0]) / dt**2
    expected[:, -1] = (data[:, -1] - 2 * data[:, -2] + data[:, -3]) / dt**2
    np.testing.assert_allclose(acc, expected, rtol=1e-12)


def test_derivative_sine_matches_analytic():
    dt = 1e-3
    w = 2 * np.pi * 5.0
    t = np.arange(2000) * dt
    data = np.sin(w * t).reshape(1, -1)
    ds = _surface(data, dt=dt)
    v = derivative(ds, DerivativeParams(order=1)).fields.read("velocity")[0]
    analytic = w * np.cos(w * t)
    # backward difference is O(dt); compare interior with a loose tol.
    np.testing.assert_allclose(v[10:-10], analytic[10:-10], atol=0.05 * w)


def test_derivative_rejects_time_aggregated_source():
    verts = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0]], dtype=np.float64)
    tris = np.array([[0, 1, 2]], dtype=np.int32)
    ds = SurfaceDataSource(
        time=TimeAxis(initial_time=0.0, timestep_size=0.0, n_timesteps=0),
        topology=Topology.triangles(tris, verts),
        elements=ElementMeta(),
        fields=MemoryFieldStore({"displacement": np.array([1.0])}),
    )
    with pytest.raises(ValueError):
        derivative(ds, DerivativeParams(order=1))


def test_second_derivative_requires_three_timesteps():
    data = np.zeros((1, 2))
    ds = _surface(data)
    with pytest.raises(ValueError):
        derivative(ds, DerivativeParams(order=2))
