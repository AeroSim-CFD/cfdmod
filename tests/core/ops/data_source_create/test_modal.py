"""Unit tests for modal_projection / modal_recomposition.

Round-trip identity: phi^T @ phi = I (mass-normalised modes) means
``modal_recomposition(modal_projection(force))`` recovers the original
force timeseries.
"""

from __future__ import annotations

import numpy as np
import pytest

from cfdmod.adapters.memory import MemoryFieldStore
from cfdmod.core import (
    ElementMeta,
    ModesDataSource,
    PointsDataSource,
    SurfaceDataSource,
    TimeAxis,
    Topology,
)
from cfdmod.core.ops.data_source_create import (
    ModalProjectionParams,
    ModalRecompositionParams,
    modal_projection,
    modal_recomposition,
)


def _surface_with_force(n_elements: int = 4, n_t: int = 6) -> SurfaceDataSource:
    verts = np.tile(np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0]]), (n_elements, 1)).astype(
        np.float64
    )
    tris = np.arange(n_elements * 3).reshape(n_elements, 3).astype(np.int32)
    f = np.arange(n_elements * n_t, dtype=np.float64).reshape(n_elements, n_t)
    return SurfaceDataSource(
        time=TimeAxis(initial_time=0.0, timestep_size=0.1, n_timesteps=n_t),
        topology=Topology.triangles(tris, verts),
        elements=ElementMeta(),
        fields=MemoryFieldStore({"force": f}),
    )


def test_modal_projection_returns_modes_data_source():
    ds = _surface_with_force(n_elements=4, n_t=5)
    phi = np.array([[1, 0], [0, 1], [1, 0], [0, 1]], dtype=np.float64)
    out = modal_projection(ds, ModalProjectionParams(mode_shapes=phi))
    assert out.kind == "modes"
    assert out.n_elements == 2
    expected = phi.T @ ds.fields.read("force")
    np.testing.assert_allclose(out.fields.read("q"), expected)


def test_modal_projection_rejects_size_mismatch():
    ds = _surface_with_force()
    phi = np.zeros((10, 2))
    with pytest.raises(ValueError):
        modal_projection(ds, ModalProjectionParams(mode_shapes=phi))


def test_modal_recomposition_round_trips_orthonormal_modes():
    ds = _surface_with_force(n_elements=4, n_t=5)
    # Orthonormal columns: phi^T @ phi = I
    rng = np.random.default_rng(0)
    phi, _ = np.linalg.qr(rng.normal(size=(4, 4)))
    modes = modal_projection(ds, ModalProjectionParams(mode_shapes=phi))
    pts = np.zeros((4, 3), dtype=np.float64)
    pts[:, 0] = np.arange(4)
    recovered = modal_recomposition(
        modes, ModalRecompositionParams(mode_shapes=phi, target_points=pts)
    )
    assert recovered.kind == "points"
    np.testing.assert_allclose(recovered.fields.read("u"), ds.fields.read("force"), atol=1e-10)


def test_modal_recomposition_requires_modes_data_source():
    ds = _surface_with_force()
    pts = np.zeros((1, 3))
    with pytest.raises(TypeError):
        modal_recomposition(
            ds,  # type: ignore[arg-type]
            ModalRecompositionParams(
                mode_shapes=np.eye(1), target_points=pts, field="force"
            ),
        )


def test_modal_projection_keeps_mode_labels():
    ds = _surface_with_force(n_elements=2)
    phi = np.array([[1.0, 0.0, 0.5], [0.0, 1.0, 0.5]])
    out = modal_projection(
        ds,
        ModalProjectionParams(mode_shapes=phi, mode_labels=["bend", "twist", "third"]),
    )
    assert out.elements.annotations["mode_labels"] == ["bend", "twist", "third"]


def test_modes_data_source_constructed_independently_of_projection():
    """Smoke check that ModesDataSource is usable on its own."""
    q = np.zeros((3, 4))
    md = ModesDataSource(
        time=TimeAxis(initial_time=0.0, timestep_size=0.1, n_timesteps=4),
        topology=None,
        elements=ElementMeta(),
        fields=MemoryFieldStore({"q": q}),
    )
    assert md.n_elements == 3
