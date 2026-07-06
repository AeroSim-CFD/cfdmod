"""Adapter conformance: MemoryStorage and XdmfH5Storage agree.

Proves the two storage backends behave identically on the same
DataSource (round-trip value equivalence), and locks the stats
round-trip regression: a time-aggregated source with bare stat fields
must read back with the same field names -- no spurious stats/Triangles
/ stats/Geometry fields and no synthetic "stats/" prefix.
"""

from __future__ import annotations

import numpy as np
import pytest

from cfdmod.adapters.memory import MemoryFieldStore, MemoryStorage
from cfdmod.adapters.xdmf_h5 import XdmfH5Storage
from cfdmod.core import ElementMeta, SurfaceDataSource, TimeAxis, Topology


def _surface_timeseries() -> SurfaceDataSource:
    verts = np.array([[0, 0, 0], [1, 0, 0], [1, 1, 0], [0, 1, 0]], dtype=np.float64)
    tris = np.array([[0, 1, 2], [0, 2, 3]], dtype=np.int32)
    cp = np.array([[0.1, 0.2, 0.3], [1.0, 1.1, 1.2]])
    return SurfaceDataSource(
        time=TimeAxis(initial_time=0.0, timestep_size=0.5, n_timesteps=3),
        topology=Topology.triangles(tris, verts),
        elements=ElementMeta(),
        fields=MemoryFieldStore({"cp": cp}),
    )


def _surface_stats() -> SurfaceDataSource:
    verts = np.array([[0, 0, 0], [1, 0, 0], [1, 1, 0], [0, 1, 0]], dtype=np.float64)
    tris = np.array([[0, 1, 2], [0, 2, 3]], dtype=np.int32)
    return SurfaceDataSource(
        time=TimeAxis(initial_time=0.0, timestep_size=0.0, n_timesteps=0),
        topology=Topology.triangles(tris, verts),
        elements=ElementMeta(),
        fields=MemoryFieldStore({"mean": np.array([0.2, 1.1]), "max": np.array([0.3, 1.2])}),
    )


def _storages(tmp_path):
    return {
        "memory": MemoryStorage(),
        "xdmf_h5": XdmfH5Storage(tmp_path),
    }


@pytest.mark.parametrize("backend", ["memory", "xdmf_h5"])
def test_surface_timeseries_round_trip(backend, tmp_path):
    ds = _surface_timeseries()
    storage = _storages(tmp_path)[backend]
    storage.write_data_source("cp_t.default", ds)
    back = storage.read_data_source("cp_t.default")

    assert back.kind == "surface"
    assert sorted(back.field_names) == ["cp"]
    assert np.allclose(back.fields.read("cp"), ds.fields.read("cp"))
    assert back.time.n_timesteps == 3
    assert np.allclose(back.time.times(), ds.time.times())


@pytest.mark.parametrize("backend", ["memory", "xdmf_h5"])
def test_stats_round_trip_field_names(backend, tmp_path):
    ds = _surface_stats()
    storage = _storages(tmp_path)[backend]
    storage.write_data_source("stats.default", ds)
    back = storage.read_data_source("stats.default")

    # Regression: no spurious geometry fields, no "stats/" prefix.
    assert sorted(back.field_names) == ["max", "mean"]
    assert np.allclose(back.fields.read("mean"), ds.fields.read("mean"))
    assert np.allclose(back.fields.read("max"), ds.fields.read("max"))
    assert back.time.is_time_aggregated


def test_backends_agree_on_timeseries(tmp_path):
    """The two backends return equal field arrays for the same source."""
    ds = _surface_timeseries()
    mem = MemoryStorage()
    h5 = XdmfH5Storage(tmp_path)
    mem.write_data_source("cp_t.default", ds)
    h5.write_data_source("cp_t.default", ds)

    a = mem.read_data_source("cp_t.default")
    b = h5.read_data_source("cp_t.default")
    assert sorted(a.field_names) == sorted(b.field_names)
    assert np.allclose(a.fields.read("cp"), b.fields.read("cp"))
    assert a.kind == b.kind
