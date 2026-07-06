"""Unit tests for the dynamic-analysis recipe."""

from __future__ import annotations

import numpy as np

from cfdmod.adapters.memory import MemoryFieldStore
from cfdmod.core import (
    ElementMeta,
    ModesDataSource,
    SurfaceDataSource,
    TimeAxis,
    Topology,
)
from cfdmod.core.recipes import (
    DynamicAnalysisConfig,
    build_dynamic_response,
    identity_solver,
)


def _surface_force(n_e: int = 3, n_t: int = 4) -> SurfaceDataSource:
    verts = np.tile(np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0]]), (n_e, 1)).astype(np.float64)
    tris = np.arange(n_e * 3).reshape(n_e, 3).astype(np.int32)
    return SurfaceDataSource(
        time=TimeAxis(initial_time=0.0, timestep_size=0.1, n_timesteps=n_t),
        topology=Topology.triangles(tris, verts),
        elements=ElementMeta(),
        fields=MemoryFieldStore(
            {"force": np.arange(n_e * n_t, dtype=np.float64).reshape(n_e, n_t)}
        ),
    )


def test_dynamic_recipe_with_identity_solver_round_trips():
    """Orthonormal modes + identity solver -> output equals input force."""
    rng = np.random.default_rng(0)
    src = _surface_force(n_e=4, n_t=5)
    phi, _ = np.linalg.qr(rng.normal(size=(4, 4)))

    target_pts = np.zeros((4, 3), dtype=np.float64)
    target_pts[:, 0] = np.arange(4)

    out = build_dynamic_response(
        src,
        DynamicAnalysisConfig(mode_shapes=phi, target_points=target_pts),
        solver=identity_solver,
    )
    np.testing.assert_allclose(out.fields.read("u"), src.fields.read("force"), atol=1e-10)


def test_dynamic_recipe_solver_can_scale_modal_response():
    src = _surface_force(n_e=3, n_t=2)
    phi = np.eye(3)
    target_pts = np.zeros((3, 3), dtype=np.float64)

    def double_solver(modes: ModesDataSource) -> ModesDataSource:
        q = modes.fields.read("q")
        return modes.with_field("q", q * 2)

    out = build_dynamic_response(
        src,
        DynamicAnalysisConfig(mode_shapes=phi, target_points=target_pts),
        solver=double_solver,
    )
    np.testing.assert_allclose(out.fields.read("u"), src.fields.read("force") * 2.0)
