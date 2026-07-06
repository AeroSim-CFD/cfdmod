"""Unit tests for the pedestrian comfort recipe."""

from __future__ import annotations

import numpy as np

from cfdmod.adapters.memory import MemoryFieldStore
from cfdmod.core import ElementMeta, PointsDataSource, TimeAxis, Topology
from cfdmod.core.recipes import PedestrianComfortConfig, build_pedestrian_comfort


def test_pedestrian_comfort_extracts_then_aggregates():
    pos = np.array([[0, 0, 0], [1, 0, 0], [2, 0, 0]], dtype=np.float64)
    rng = np.random.default_rng(0)
    u = rng.normal(loc=2.0, size=(3, 200))
    src = PointsDataSource(
        time=TimeAxis(initial_time=0.0, timestep_size=0.1, n_timesteps=200),
        topology=Topology.points(pos),
        elements=ElementMeta(position=pos),
        fields=MemoryFieldStore({"u_mag": u.astype(np.float64)}),
    )
    out = build_pedestrian_comfort(src, PedestrianComfortConfig(probes=np.array([[0, 0, 0]])))
    assert out.kind == "points"
    assert out.time.is_time_aggregated
    assert "mean" in out.field_names
    assert abs(out.fields.read("mean")[0] - 2.0) < 0.5
