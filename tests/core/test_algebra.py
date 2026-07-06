"""Unit tests for the four broadcasting rules in :mod:`cfdmod.core.algebra`."""

from __future__ import annotations

import numpy as np
import pytest

from cfdmod.adapters.memory import MemoryFieldStore
from cfdmod.core import (
    ElementMeta,
    SurfaceDataSource,
    TimeAxis,
    Topology,
)
from cfdmod.core.algebra import add, classify_broadcast, div, mul, sub


def _surface(field: np.ndarray, n_timesteps: int) -> SurfaceDataSource:
    n_elements = field.shape[0]
    verts = np.array([[i, 0, 0] for i in range(n_elements + 2)], dtype=np.float64)
    tris = np.array([[i, i + 1, i + 2] for i in range(n_elements)], dtype=np.int32)
    if n_timesteps == 0:
        time = TimeAxis(initial_time=0.0, timestep_size=0.0, n_timesteps=0)
    else:
        time = TimeAxis(initial_time=0.0, timestep_size=0.1, n_timesteps=n_timesteps)
    return SurfaceDataSource(
        time=time,
        topology=Topology.triangles(tris, verts),
        elements=ElementMeta(),
        fields=MemoryFieldStore({"p": field}),
    )


def test_classify_constant():
    assert classify_broadcast((10, 4), None) == "constant"
    assert classify_broadcast((10,), None) == "constant"


def test_classify_elementwise():
    assert classify_broadcast((10, 4), (10, 4)) == "elementwise"


def test_classify_column():
    assert classify_broadcast((10, 4), (1, 4)) == "column"


def test_classify_row():
    assert classify_broadcast((10, 4), (10,)) == "row"
    assert classify_broadcast((10,), (10, 4)) == "row"


def test_classify_rejects_unmatched_shapes():
    with pytest.raises(ValueError):
        classify_broadcast((10, 4), (3, 4))


def test_constant_rule_uniform_scaling():
    p = np.arange(12, dtype=np.float64).reshape(3, 4)
    ds = _surface(p, n_timesteps=4)
    out = mul(ds, 2.0, field="p", out="p_scaled")
    assert np.allclose(out.fields.read("p_scaled"), p * 2.0)


def test_column_rule_subtracts_reference_per_timestep():
    p = np.arange(12, dtype=np.float64).reshape(3, 4)
    p_ref = np.array([[10.0, 20.0, 30.0, 40.0]])  # shape (1, 4)
    lhs = _surface(p, n_timesteps=4)
    # rhs uses the same time axis but a single-element topology
    rhs_verts = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0]], dtype=np.float64)
    rhs = SurfaceDataSource(
        time=lhs.time,
        topology=Topology.triangles([[0, 1, 2]], rhs_verts),
        elements=ElementMeta(),
        fields=MemoryFieldStore({"p": p_ref}),
    )
    out = sub(lhs, rhs, field="p", out="cp")
    assert out.fields.shape("cp") == (3, 4)
    assert np.allclose(out.fields.read("cp"), p - p_ref)


def test_row_rule_one_side_time_aggregated():
    p = np.arange(12, dtype=np.float64).reshape(3, 4)
    profile = np.array([1.0, 2.0, 4.0])  # shape (3,)
    lhs = _surface(p, n_timesteps=4)
    rhs_verts = lhs.topology.vertices
    rhs_tris = lhs.topology.connectivity
    rhs = SurfaceDataSource(
        time=TimeAxis(initial_time=0.0, timestep_size=0.0, n_timesteps=0),
        topology=Topology.triangles(rhs_tris, rhs_verts),
        elements=ElementMeta(),
        fields=MemoryFieldStore({"p": profile}),
    )
    out = div(lhs, rhs, field="p", out="ratio")
    assert np.allclose(out.fields.read("ratio"), p / profile[:, None])


def test_elementwise_rule_same_shape():
    a = np.arange(12, dtype=np.float64).reshape(3, 4)
    b = np.full_like(a, 0.5)
    lhs = _surface(a, n_timesteps=4)
    rhs = _surface(b, n_timesteps=4)
    out = add(lhs, rhs, field="p")
    assert np.allclose(out.fields.read("p"), a + b)


def test_default_out_field_overwrites_input():
    a = np.ones((2, 3))
    ds = _surface(a, n_timesteps=3)
    out = add(ds, 1.0, field="p")
    assert np.allclose(out.fields.read("p"), 2.0)
    # Predecessor is unaffected
    assert np.allclose(ds.fields.read("p"), 1.0)
