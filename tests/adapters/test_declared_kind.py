"""A caller that knows the DataSource kind must be able to say so.

The XDMF+H5 byte layout does not record whether a file is a surface or a
points source, so the adapter used to guess from the filename stem: anything
not named ``points.*`` read back as a surface. That is a convention a human
can follow and a caller generating filenames from an id cannot -- and a
template already declares ``kind:`` per input, so the information was there
and being thrown away.
"""

from __future__ import annotations

import numpy as np
import pytest

from cfdmod.adapters.memory import MemoryFieldStore, MemoryStorage
from cfdmod.adapters.xdmf_h5 import XdmfH5Storage
from cfdmod.core import ElementMeta, PointsDataSource, SurfaceDataSource, TimeAxis, Topology
from cfdmod.core.errors import TemplateError
from cfdmod.core.pipeline_yaml import PipelineTemplate, run_template

pytestmark = pytest.mark.unit


def _points() -> PointsDataSource:
    coords = np.array([[0.0, 0.0, 1.0], [0.0, 0.0, 2.0]])
    return PointsDataSource(
        time=TimeAxis(initial_time=0.0, timestep_size=0.5, n_timesteps=3),
        topology=Topology.points(coords),
        elements=ElementMeta(),
        fields=MemoryFieldStore({"pressure": np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])}),
    )


def _surface() -> SurfaceDataSource:
    verts = np.array([[0, 0, 0], [1, 0, 0], [1, 1, 0]], dtype=np.float64)
    tris = np.array([[0, 1, 2]], dtype=np.int32)
    return SurfaceDataSource(
        time=TimeAxis(initial_time=0.0, timestep_size=0.5, n_timesteps=3),
        topology=Topology.triangles(tris, verts),
        elements=ElementMeta(),
        fields=MemoryFieldStore({"pressure": np.array([[1.0, 2.0, 3.0]])}),
    )


def test_h5_infers_points_only_from_the_filename(tmp_path):
    """The defect, pinned: without a declared kind the stem is all there is."""
    storage = XdmfH5Storage(tmp_path, write_xdmf=False)
    storage.write_data_source("probe_17", _points())

    assert storage.read_data_source("probe_17").kind == "surface"


def test_h5_declared_kind_beats_the_filename(tmp_path):
    """...and declaring it fixes exactly that."""
    storage = XdmfH5Storage(tmp_path, write_xdmf=False)
    storage.write_data_source("probe_17", _points())

    back = storage.read_data_source("probe_17", kind="points")
    assert back.kind == "points"
    assert back.topology.cell_type == "point"
    np.testing.assert_allclose(back.fields.read("pressure"), [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])


def test_h5_declared_surface_under_a_points_name(tmp_path):
    """The mirror case: a surface whose stem happens to start with points."""
    storage = XdmfH5Storage(tmp_path, write_xdmf=False)
    storage.write_data_source("points.actually_a_surface", _surface())

    assert storage.read_data_source("points.actually_a_surface").kind == "points"
    back = storage.read_data_source("points.actually_a_surface", kind="surface")
    assert back.kind == "surface"
    assert back.topology.cell_type == "triangle"


@pytest.mark.parametrize("kind", ["volume", "modes"])
def test_h5_rejects_a_kind_it_cannot_represent(tmp_path, kind):
    """Better a refusal than a surface handed back under another name."""
    storage = XdmfH5Storage(tmp_path, write_xdmf=False)
    storage.write_data_source("thing", _surface())

    with pytest.raises(ValueError, match="cannot read"):
        storage.read_data_source("thing", kind=kind)


def test_memory_storage_validates_the_declared_kind():
    """Both backends must agree, or a test passing on memory means nothing."""
    storage = MemoryStorage()
    storage.write_data_source("probe", _points())

    assert storage.read_data_source("probe", kind="points").kind == "points"
    with pytest.raises(ValueError, match="but 'surface' was requested"):
        storage.read_data_source("probe", kind="surface")


def test_run_template_passes_the_declared_kind_down(tmp_path):
    """The end-to-end case the issue is about.

    A points input under a filename that carries no ``points.`` prefix used to
    fail the kind assertion; declaring the kind is now enough.
    """
    storage = XdmfH5Storage(tmp_path, write_xdmf=False)
    storage.write_data_source("probe_17", _points())

    template = PipelineTemplate.model_validate(
        {
            "name": "kind_passthrough",
            "root": str(tmp_path),
            "inputs": {"p_ref": {"kind": "points", "path": "probe_17", "field": "pressure"}},
            "pipeline": [],
            "outputs": {},
        }
    )
    bindings = run_template(template, storage=storage)
    assert bindings["p_ref"].kind == "points"


def test_run_template_reports_an_unreadable_declared_kind(tmp_path):
    storage = XdmfH5Storage(tmp_path, write_xdmf=False)
    storage.write_data_source("thing", _surface())

    template = PipelineTemplate.model_validate(
        {
            "name": "bad_kind",
            "root": str(tmp_path),
            "inputs": {"v": {"kind": "volume", "path": "thing"}},
            "pipeline": [],
            "outputs": {},
        }
    )
    with pytest.raises(TemplateError, match="cannot read"):
        run_template(template, storage=storage)


def test_run_template_tolerates_a_backend_without_the_keyword():
    """``Storage`` is structural: an adapter on the old signature still works."""

    class LegacyStorage:
        def __init__(self, inner):
            self._inner = inner

        def read_data_source(self, key):  # no ``kind`` keyword, on purpose
            return self._inner.read_data_source(key)

        def write_data_source(self, key, ds):
            self._inner.write_data_source(key, ds)

        def keys(self):
            return self._inner.keys()

    inner = MemoryStorage()
    inner.write_data_source("probe", _points())
    template = PipelineTemplate.model_validate(
        {
            "name": "legacy",
            "inputs": {"p_ref": {"kind": "points", "path": "probe"}},
            "pipeline": [],
            "outputs": {},
        }
    )
    bindings = run_template(template, storage=LegacyStorage(inner))
    assert bindings["p_ref"].kind == "points"
