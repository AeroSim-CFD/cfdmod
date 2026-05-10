"""Unit tests for :func:`compose` and :func:`identity`."""

from __future__ import annotations

from functools import partial

import numpy as np

from cfdmod.adapters.memory import MemoryFieldStore
from cfdmod.core import (
    ElementMeta,
    SurfaceDataSource,
    TimeAxis,
    Topology,
)
from cfdmod.core.algebra import add, mul
from cfdmod.core.pipeline import compose, identity


def _ds() -> SurfaceDataSource:
    verts = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0]], dtype=np.float64)
    tris = np.array([[0, 1, 2]], dtype=np.int32)
    return SurfaceDataSource(
        time=TimeAxis(initial_time=0.0, timestep_size=1.0, n_timesteps=3),
        topology=Topology.triangles(tris, verts),
        elements=ElementMeta(),
        fields=MemoryFieldStore({"p": np.array([[1.0, 2.0, 3.0]])}),
    )


def test_identity_returns_same_instance():
    ds = _ds()
    assert identity(ds) is ds


def test_compose_runs_left_to_right():
    pipe = compose(
        partial(add, rhs=1.0, field="p"),  # 1+1=2, 2+1=3, 3+1=4
        partial(mul, rhs=2.0, field="p"),  # 4, 6, 8
    )
    out = pipe(_ds())
    assert np.allclose(out.fields.read("p"), [[4.0, 6.0, 8.0]])


def test_compose_with_no_args_returns_identity():
    pipe = compose()
    ds = _ds()
    assert pipe(ds) is ds
