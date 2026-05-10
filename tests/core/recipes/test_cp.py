"""Unit tests for the Cp recipe."""

from __future__ import annotations

import numpy as np
import pytest

from cfdmod.adapters.memory import MemoryFieldStore
from cfdmod.core import ElementMeta, SurfaceDataSource, TimeAxis, Topology
from cfdmod.core.recipes import CpRecipeConfig, build_cp, cp_pipeline


def _surface(values: np.ndarray, dt: float = 0.1) -> SurfaceDataSource:
    n_elements, n_t = values.shape
    verts = np.tile(np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0]]), (n_elements, 1)).astype(
        np.float64
    )
    tris = np.arange(n_elements * 3).reshape(n_elements, 3).astype(np.int32)
    return SurfaceDataSource(
        time=TimeAxis(initial_time=0.0, timestep_size=dt, n_timesteps=n_t),
        topology=Topology.triangles(tris, verts),
        elements=ElementMeta(),
        fields=MemoryFieldStore({"pressure": values.astype(np.float64)}),
    )


def test_cp_with_scalar_reference_and_constant_q():
    body = _surface(np.full((3, 5), 100.0))
    out = build_cp(body, p_ref=20.0, cfg=CpRecipeConfig(dynamic_pressure=10.0))
    assert "cp" in out.field_names
    assert "pressure" in out.field_names  # original retained
    np.testing.assert_allclose(out.fields.read("cp"), 8.0)


def test_cp_with_reference_data_source_column_wise():
    body = _surface(np.tile(np.arange(5, dtype=np.float64), (3, 1)) + 100.0)
    ref_arr = np.arange(5, dtype=np.float64).reshape(1, 5)  # shape (1, 5) -> column rule
    ref_verts = np.array([[0, 0, 0]], dtype=np.float64)
    p_ref = SurfaceDataSource(
        time=body.time,
        topology=Topology.triangles(np.array([[0, 0, 0]], dtype=np.int32), ref_verts),
        elements=ElementMeta(),
        fields=MemoryFieldStore({"pressure": ref_arr}),
    )
    out = build_cp(body, p_ref=p_ref, cfg=CpRecipeConfig(dynamic_pressure=2.0))
    expected = (body.fields.read("pressure") - ref_arr) / 2.0
    np.testing.assert_allclose(out.fields.read("cp"), expected)


def test_cp_with_time_rescale():
    body = _surface(np.full((1, 4), 50.0), dt=0.5)
    out = build_cp(
        body,
        p_ref=10.0,
        cfg=CpRecipeConfig(dynamic_pressure=4.0, time_rescale_factor=2.0),
    )
    assert out.time.timestep_size == pytest.approx(1.0)
    np.testing.assert_allclose(out.fields.read("cp"), 10.0)


def test_cp_with_statistics_collapses_time_axis():
    rng = np.random.default_rng(0)
    body = _surface(rng.normal(size=(4, 200)) * 10.0 + 100.0)
    out = build_cp(
        body,
        p_ref=100.0,
        cfg=CpRecipeConfig(dynamic_pressure=10.0, statistics=["mean", "rms"]),
    )
    assert out.time.is_time_aggregated
    assert sorted(out.field_names) == ["mean", "rms"]
    # cp = (p - 100) / 10; with p ~ N(100, 10) -> cp ~ N(0, 1).
    assert abs(out.fields.read("mean").mean()) < 0.1
    assert abs(out.fields.read("rms").mean() - 1.0) < 0.1


def test_cp_pipeline_is_a_callable():
    body = _surface(np.full((1, 3), 30.0))
    pipe = cp_pipeline(CpRecipeConfig(dynamic_pressure=10.0), p_ref=10.0)
    assert callable(pipe)
    out = pipe(body)
    np.testing.assert_allclose(out.fields.read("cp"), 2.0)
