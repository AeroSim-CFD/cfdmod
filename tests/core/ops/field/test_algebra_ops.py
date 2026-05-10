"""Unit tests for the field algebra ops (op-shaped wrappers)."""

from __future__ import annotations

import numpy as np

from cfdmod.adapters.memory import MemoryFieldStore
from cfdmod.core import ElementMeta, SurfaceDataSource, TimeAxis, Topology
from cfdmod.core.ops.field.algebra import (
    ScaleParams,
    SubParams,
    scale,
    sub,
)


def _surface(values: np.ndarray) -> SurfaceDataSource:
    n_elements, n_t = values.shape
    verts = np.tile(np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0]]), (n_elements, 1)).astype(
        np.float64
    )
    tris = np.arange(n_elements * 3).reshape(n_elements, 3).astype(np.int32)
    return SurfaceDataSource(
        time=TimeAxis(initial_time=0.0, timestep_size=1.0, n_timesteps=n_t),
        topology=Topology.triangles(tris, verts),
        elements=ElementMeta(),
        fields=MemoryFieldStore({"pressure": values.astype(np.float64)}),
    )


def test_scale_field_constant_broadcast():
    ds = _surface(np.full((2, 3), 5.0))
    out = scale(ds, ScaleParams(field="pressure", factor=4.0))
    np.testing.assert_array_equal(out.fields.read("pressure"), 20.0)


def test_sub_field_with_data_source():
    ds_lhs = _surface(np.full((2, 3), 10.0))
    ds_rhs = _surface(np.full((2, 3), 4.0))
    out = sub(ds_lhs, ds_rhs, SubParams(field="pressure", out="dp"))
    np.testing.assert_array_equal(out.fields.read("dp"), 6.0)
    np.testing.assert_array_equal(out.fields.read("pressure"), 10.0)
