"""Unit tests for the attach_grouping geometric op."""

from __future__ import annotations

import numpy as np
import pytest

from cfdmod.adapters.memory import MemoryFieldStore
from cfdmod.core import (
    ElementMeta,
    Grouping,
    SurfaceDataSource,
    TimeAxis,
    Topology,
)
from cfdmod.core.ops.geometric import AttachGroupingParams, attach_grouping


def _surface(n: int = 3) -> SurfaceDataSource:
    verts = np.tile(np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0]]), (n, 1)).astype(np.float64)
    tris = np.arange(n * 3).reshape(n, 3).astype(np.int32)
    return SurfaceDataSource(
        time=TimeAxis(initial_time=0.0, timestep_size=0.1, n_timesteps=2),
        topology=Topology.triangles(tris, verts),
        elements=ElementMeta(),
        fields=MemoryFieldStore({"pressure": np.zeros((n, 2))}),
    )


def test_attach_grouping_adds_new_grouping():
    ds = _surface(3)
    g = Grouping(name="body", indices=[0, 0, 1])
    out = attach_grouping(ds, AttachGroupingParams(grouping=g))
    assert "body" in out.groupings
    assert ds.groupings == {}  # original unchanged


def test_attach_grouping_rejects_size_mismatch():
    ds = _surface(3)
    g = Grouping(name="body", indices=[0, 0, 0, 0])
    with pytest.raises(ValueError):
        attach_grouping(ds, AttachGroupingParams(grouping=g))
