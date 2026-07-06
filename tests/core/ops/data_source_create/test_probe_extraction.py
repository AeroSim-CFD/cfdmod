"""Unit tests for probe_extraction."""

from __future__ import annotations

import numpy as np

from cfdmod.adapters.memory import MemoryFieldStore
from cfdmod.core import ElementMeta, PointsDataSource, TimeAxis, Topology
from cfdmod.core.ops.data_source_create import ProbeExtractionParams, probe_extraction


def _column(z: np.ndarray, values: np.ndarray) -> PointsDataSource:
    pos = np.zeros((z.size, 3))
    pos[:, 2] = z
    return PointsDataSource(
        time=TimeAxis(initial_time=0.0, timestep_size=1.0, n_timesteps=values.shape[1]),
        topology=Topology.points(pos),
        elements=ElementMeta(position=pos),
        fields=MemoryFieldStore({"u": values.astype(np.float64)}),
    )


def test_nearest_picks_closest_source_element():
    src = _column(np.array([0.0, 1.0, 2.0]), np.array([[10.0], [20.0], [30.0]]))
    out = probe_extraction(src, ProbeExtractionParams(probes=np.array([[0, 0, 0.4]]), field="u"))
    np.testing.assert_array_equal(out.fields.read("u"), [[10.0]])


def test_linear_zaxis_interpolates_between_samples():
    src = _column(np.array([0.0, 1.0, 2.0]), np.array([[0.0], [10.0], [20.0]]))
    out = probe_extraction(
        src,
        ProbeExtractionParams(probes=np.array([[0, 0, 0.5]]), field="u", mode="linear_zaxis"),
    )
    np.testing.assert_allclose(out.fields.read("u"), [[5.0]])


def test_extracted_source_keeps_time_axis():
    src = _column(
        np.array([0.0, 1.0, 2.0]),
        np.tile(np.arange(4, dtype=np.float64), (3, 1)),
    )
    out = probe_extraction(
        src,
        ProbeExtractionParams(probes=np.array([[0, 0, 1.0]]), field="u"),
    )
    assert out.time.n_timesteps == 4
    np.testing.assert_array_equal(out.fields.read("u")[0], np.arange(4))
