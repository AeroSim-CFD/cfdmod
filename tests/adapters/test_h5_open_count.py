"""Guard that h5 writes do not reopen the file once per timestep.

Every write helper in ``cfdmod.io.xdmf`` opens and closes the file per call.
That is correct for a one-off and quietly expensive in a loop: writing a
timeseries used to cost one full open / flush / close cycle *per timestep*,
for output that is byte-for-byte identical to writing it through one handle.

Counting opens rather than timing the write is deliberate. A timing assertion
on a shared machine flakes; the open count is exact, and it is the property
that actually matters - the write must scale with the data, not with the number
of timesteps.
"""

from __future__ import annotations

import numpy as np
import pytest

import cfdmod.io.xdmf as _xdmf
from cfdmod.adapters import XdmfH5Storage
from cfdmod.adapters.memory import MemoryFieldStore
from cfdmod.core import SurfaceDataSource, TimeAxis
from cfdmod.core.topology import ElementMeta, Topology

pytestmark = pytest.mark.unit


@pytest.fixture()
def count_h5_opens(monkeypatch):
    """Return a list that accumulates one entry per ``h5py.File`` construction."""
    opens: list[str] = []
    real_file = _xdmf.h5py.File

    def counting_file(path, *args, **kwargs):
        opens.append(str(path))
        return real_file(path, *args, **kwargs)

    monkeypatch.setattr(_xdmf.h5py, "File", counting_file)
    return opens


def _surface(n_elements: int, n_timesteps: int) -> SurfaceDataSource:
    vertices = np.random.default_rng(0).random((n_elements * 3, 3))
    triangles = np.arange(n_elements * 3, dtype=np.int32).reshape(n_elements, 3)
    return SurfaceDataSource(
        time=TimeAxis(initial_time=0.0, timestep_size=0.1, n_timesteps=n_timesteps),
        topology=Topology.triangles(triangles, vertices),
        elements=ElementMeta(),
        fields=MemoryFieldStore(
            {"cp": np.random.default_rng(1).random((n_elements, n_timesteps))}
        ),
    )


@pytest.mark.parametrize("n_timesteps", [4, 64])
def test_write_data_source_open_count_is_independent_of_timesteps(
    tmp_path, count_h5_opens, n_timesteps
):
    storage = XdmfH5Storage(tmp_path, write_xdmf=False)
    storage.write_data_source(f"cp_t.n{n_timesteps}", _surface(6, n_timesteps))
    # One handle for the whole write: geometry, meta and every field column.
    assert len(count_h5_opens) == 1


def test_write_data_source_still_writes_every_timestep(tmp_path):
    """The open-count guard must not be satisfiable by writing less data."""
    storage = XdmfH5Storage(tmp_path, write_xdmf=False)
    ds = _surface(6, 16)
    storage.write_data_source("cp_t.full", ds)

    back = storage.read_data_source("cp_t.full")
    assert back.time.n_timesteps == 16
    np.testing.assert_allclose(back.fields.read("cp"), ds.fields.read("cp"))


def test_regroup_open_count_is_independent_of_timesteps(tmp_path, count_h5_opens):
    """The regroup writer streams step by step; it must still hold both handles.

    Regroup reopened *two* files per timestep - the input to read the column and
    the output to write it - so it paid double.
    """
    import cfdmod.regroup.functions as regroup_functions

    n_timesteps = 32
    src = tmp_path / "in.h5"
    triangles = np.array([[0, 1, 2], [3, 4, 5]], dtype=np.int32)
    vertices = np.random.default_rng(2).random((6, 3))
    _xdmf.write_timeseries_geometry(src, triangles, vertices)
    times = np.arange(n_timesteps, dtype=np.float64)
    _xdmf.write_timeseries_meta(src, times, times)
    with _xdmf.timeseries_writer(src) as w:
        for t in times:
            w.write_step("cp", f"t{t}", np.array([1.0 + t, 2.0 + t]))

    index = regroup_functions.RegroupIndex(
        new_to_parent=np.array([0, 1, 0], dtype=np.int64),
        output_group_names=["a"],
        triangle_group_idx=np.array([0, 0, 0], dtype=np.int64),
        group_parents=[np.array([0, 1], dtype=np.int64)],
        group_weights=[np.array([0.5, 0.5])],
        aggregation="per_triangle",
    )
    new_triangles = np.array([[0, 1, 2], [3, 4, 5], [0, 1, 2]], dtype=np.int32)

    count_h5_opens.clear()
    regroup_functions.apply_regroup_to_timeseries(
        input_h5=src,
        output_h5=tmp_path / "out.h5",
        regroup_index=index,
        new_triangles=new_triangles,
        new_vertices=vertices,
        group="cp",
    )
    # Bounded by the fixed set-up reads/writes (geometry, meta, key listing),
    # never by n_timesteps. Before the fix this was 2 * n_timesteps + set-up.
    assert len(count_h5_opens) < n_timesteps
