import pathlib

import h5py
import numpy as np
import pytest
from lnas import LnasFormat

from cfdmod.io.mesh import load_mesh, mesh_from_h5
from cfdmod.io.xdmf import write_timeseries_geometry

pytestmark = pytest.mark.unit


@pytest.fixture()
def triangles():
    return np.array([[0, 1, 2], [1, 3, 2]], dtype=np.int32)


@pytest.fixture()
def vertices():
    return np.array(
        [[0, 0, 0], [0, 1, 0], [1, 0, 0], [1, 1, 0]], dtype=np.float64
    )


def test_mesh_from_h5_synthetic_surface(tmp_path, triangles, vertices):
    h5 = tmp_path / "body.h5"
    write_timeseries_geometry(h5, triangles, vertices)

    mesh = mesh_from_h5(h5)
    assert isinstance(mesh, LnasFormat)
    assert list(mesh.surfaces.keys()) == ["all"]
    assert len(mesh.geometry.triangles) == len(triangles)
    np.testing.assert_array_equal(mesh.geometry.triangles, triangles)
    np.testing.assert_array_equal(mesh.geometry.vertices, vertices)


def test_mesh_from_h5_rejects_file_without_geometry(tmp_path):
    h5 = tmp_path / "no_geom.h5"
    with h5py.File(h5, "w") as f:
        f.create_dataset("foo", data=np.zeros(3))
    with pytest.raises(ValueError, match="no /Triangles \\+ /Geometry"):
        mesh_from_h5(h5)


def test_load_mesh_returns_lnasformat_unchanged(tmp_path, triangles, vertices):
    h5 = tmp_path / "body.h5"
    write_timeseries_geometry(h5, triangles, vertices)
    pre_loaded = mesh_from_h5(h5)
    assert load_mesh(pre_loaded) is pre_loaded


def test_load_mesh_dispatches_h5(tmp_path, triangles, vertices):
    h5 = tmp_path / "body.h5"
    write_timeseries_geometry(h5, triangles, vertices)
    mesh = load_mesh(h5)
    assert list(mesh.surfaces) == ["all"]


def test_load_mesh_xdmf_redirects_to_sibling_h5(tmp_path, triangles, vertices):
    h5 = tmp_path / "body.h5"
    xdmf = tmp_path / "body.xdmf"
    write_timeseries_geometry(h5, triangles, vertices)
    xdmf.write_text("<Xdmf/>")  # contents irrelevant; reader follows the .h5
    mesh = load_mesh(xdmf)
    assert list(mesh.surfaces) == ["all"]
    np.testing.assert_array_equal(mesh.geometry.triangles, triangles)


def test_load_mesh_xdmf_missing_h5_raises(tmp_path):
    xdmf = tmp_path / "lonely.xdmf"
    xdmf.write_text("<Xdmf/>")
    with pytest.raises(FileNotFoundError):
        load_mesh(xdmf)


def test_load_mesh_lnas_preserves_named_surfaces(tmp_path):
    fixture = pathlib.Path("fixtures/tests/pressure/galpao/galpao.normalized.lnas")
    if not fixture.exists():
        pytest.skip("galpao fixture not available")
    mesh = load_mesh(fixture)
    assert "all" not in mesh.surfaces
    assert len(mesh.surfaces) > 1


def test_load_mesh_unknown_extension(tmp_path):
    bad = tmp_path / "mesh.weird"
    bad.write_bytes(b"")
    with pytest.raises(ValueError, match="Unsupported mesh format"):
        load_mesh(bad)
