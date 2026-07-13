import numpy as np
import pytest
from lnas import LnasGeometry

from cfdmod import mesh_field

pytestmark = pytest.mark.unit


def _box_geometry() -> LnasGeometry:
    """A small closed unit box as an LnasGeometry (8 vertices, 12 triangles)."""
    vertices = np.array(
        [
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [1.0, 1.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
            [1.0, 0.0, 1.0],
            [1.0, 1.0, 1.0],
            [0.0, 1.0, 1.0],
        ],
        dtype=np.float64,
    )
    triangles = np.array(
        [
            [0, 2, 1],
            [0, 3, 2],
            [4, 5, 6],
            [4, 6, 7],
            [0, 1, 5],
            [0, 5, 4],
            [1, 2, 6],
            [1, 6, 5],
            [2, 3, 7],
            [2, 7, 6],
            [3, 0, 4],
            [3, 4, 7],
        ],
        dtype=np.int64,
    )
    return LnasGeometry(vertices, triangles)


def test_slice_and_render_smoke(tmp_path):
    pytest.importorskip("pyvista")

    geom = _box_geometry()
    centroids = mesh_field.triangle_centroids(geom)
    field = centroids[:, 2]  # z-graded per-triangle field

    y0 = 0.5
    slc = mesh_field.slice_field_on_plane(
        geom, field, origin=(0.5, y0, 0.5), normal=(0.0, 1.0, 0.0)
    )

    assert slc is not None
    assert slc.points.shape[1] == 3
    assert slc.points.size > 0
    assert slc.values.shape[0] == slc.points.shape[0]
    assert np.allclose(slc.points[:, 1], y0, atol=1e-6)

    image_path = tmp_path / "slice.png"
    ok = mesh_field.render_plane_slice(slc, image_path, plane_axes=("x", "z"))
    assert ok is True
    assert image_path.exists()


def test_slice_field_on_plane_noop_without_pyvista(monkeypatch, capsys):
    monkeypatch.setattr(mesh_field, "has_pyvista", lambda: False)

    geom = _box_geometry()
    field = np.zeros(geom.triangles.shape[0], dtype=np.float64)

    result = mesh_field.slice_field_on_plane(
        geom, field, origin=(0.5, 0.5, 0.5), normal=(0.0, 1.0, 0.0)
    )
    assert result is None
    out = capsys.readouterr().out
    assert "PyVista not installed" in out


def test_render_plane_slice_noop_on_none(capsys):
    assert mesh_field.render_plane_slice(None, "unused.png") is False
    out = capsys.readouterr().out
    assert "empty slice" in out
