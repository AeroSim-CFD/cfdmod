import numpy as np
from lnas import LnasGeometry

from cfdmod.api.geometry.region_meshing import (
    create_regions_mesh,
    slice_surface,
    slice_triangle,
    triangulate_tri,
)
from cfdmod.use_cases.pressure.zoning.zoning_model import ZoningModel


def test_triangulate_tri():
    vertices = np.array([[0, 0, 0], [0, 1, 0], [1, 0, 0]], dtype=np.float32)
    single_slice_verts = np.insert(vertices, 1, [0, 0.5, 0], axis=0)
    double_slice_verts = np.insert(single_slice_verts, 3, [0.5, 0.5, 0], axis=0)
    single_slice_result = triangulate_tri(single_slice_verts, [1])
    double_slice_result = triangulate_tri(double_slice_verts, [1, 3])

    assert len(single_slice_result) == 2  # Two triangles
    assert len(double_slice_result) == 3  # Three triangles


def test_slice_triangle():
    vertices = np.array([[0, 0, 0], [0.5, 1, 0], [1, 0, 0]], dtype=np.float32)
    single_slice_result = slice_triangle(vertices, 0, 0.5)
    double_slice_result = slice_triangle(vertices, 1, 0.5)

    assert len(single_slice_result) == 2  # Two sliced triangles
    assert len(double_slice_result) == 3  # Three sliced triangles


def test_slice_surface():
    vertices = np.array([[0, 0, 0], [0, 10, 0], [10, 0, 0], [10, 10, 0]], dtype=np.float32)
    triangles = np.array([[0, 1, 2], [1, 3, 2]])
    mock_mesh = LnasGeometry(vertices, triangles)
    sliced_mesh = slice_surface(mock_mesh, 1, 5)

    assert len(sliced_mesh.vertices) == 7
    assert len(sliced_mesh.triangles) == 6


def test_multiple_slices():
    vertices = np.array([[0, 0, 0], [0, 10, 0], [10, 0, 0], [10, 10, 0]], dtype=np.float32)
    triangles = np.array([[0, 1, 2], [1, 3, 2]])
    mock_mesh = LnasGeometry(vertices, triangles)
    sliced_mesh = mock_mesh.copy()

    for x_int in np.linspace(0, 10, 10):
        sliced_mesh = slice_surface(sliced_mesh, 0, x_int)

    for y_int in np.linspace(0, 10, 10):
        sliced_mesh = slice_surface(sliced_mesh, 1, y_int)

    assert len(sliced_mesh.vertices) == 456
    assert len(sliced_mesh.triangles) == 618


def test_create_regions_mesh():
    vertices = np.array([[0, 0, 0], [0, 10, 0], [10, 0, 0], [10, 10, 0]], dtype=np.float32)
    triangles = np.array([[0, 1, 2], [1, 3, 2]])
    mock_mesh = LnasGeometry(vertices, triangles)
    zoning = ZoningModel(x_intervals=[0, 5, 10], y_intervals=[0, 10], z_intervals=[0, 10])
    zoning = zoning.offset_limits(0.1)
    region_mesh = create_regions_mesh(
        mock_mesh, (zoning.x_intervals, zoning.y_intervals, zoning.z_intervals)
    )

    assert len(region_mesh.vertices) == 7
    assert len(region_mesh.triangles) == 6
