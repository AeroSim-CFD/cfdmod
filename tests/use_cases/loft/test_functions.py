import matplotlib.tri as tri
import numpy as np
import pytest

from cfdmod.use_cases.loft.functions import (
    find_border,
    generate_circular_loft_vertices,
    generate_loft_triangles,
    get_angle_between,
    project_border,
)


@pytest.fixture()
def nx():
    yield 3


@pytest.fixture()
def ny():
    yield 3


@pytest.fixture()
def triangle_vertices(nx, ny):
    x = np.linspace(-10, 10, nx + 1)
    y = np.linspace(-10, 10, ny + 1)
    z = np.array([1])
    xv, yv, zv = np.meshgrid(x, y, z)

    vertices = np.vstack((xv.flatten(), yv.flatten(), zv.flatten())).T
    triangles = tri.Triangulation(vertices[:, 0], vertices[:, 1]).triangles

    yield vertices[triangles]


def test_find_border(nx, ny, triangle_vertices):
    border_verts, border_edges = find_border(triangle_vertices=triangle_vertices)
    expected_edge_count = (nx + ny) * 2
    expected_vertex_count = (nx + ny + 2) * 2 - 4

    assert len(border_edges) == expected_edge_count
    assert len(border_verts) == expected_vertex_count


def test_angle_between():
    vec1 = np.array([1, 0, 0])
    vec2 = np.array([0, 1, 0])
    vec3 = np.array([1, 1, 0])
    vec4 = np.array([-1, -1, 0])

    assert get_angle_between(vec1, vec2) == 90
    assert get_angle_between(vec1, vec3) == 45
    assert get_angle_between(vec1, vec4) == 225
    assert get_angle_between(vec2, vec3) == 315
    assert get_angle_between(vec3, vec4) == 180


def test_project_border(triangle_vertices):
    border_verts, _ = find_border(triangle_vertices=triangle_vertices)
    border_profile, _ = project_border(border_verts, projection_diretion=np.array([1, 0, 0]))
    assert all(border_profile[:, 0] >= 0)
    border_profile, _ = project_border(border_verts, projection_diretion=np.array([-1, 0, 0]))
    assert all(border_profile[:, 0] <= 0)
    border_profile, _ = project_border(border_verts, projection_diretion=np.array([0, 1, 0]))
    assert all(border_profile[:, 1] >= 0)
    border_profile, _ = project_border(border_verts, projection_diretion=np.array([0, -1, 0]))
    assert all(border_profile[:, 1] <= 0)


def test_loft_surface(triangle_vertices):
    projection_direction = np.array([1, 0, 0])
    border_verts, _ = find_border(triangle_vertices=triangle_vertices)
    border_profile, center = project_border(border_verts, projection_diretion=projection_direction)
    loft_verts = generate_circular_loft_vertices(
        border_profile=border_profile,
        projection_diretion=projection_direction,
        loft_length=100,
        loft_z_pos=1,
        mesh_center=center,
    )
    loft_tri, loft_normals = generate_loft_triangles(
        border_profile=border_profile, loft_vertices=loft_verts
    )

    assert len(border_profile) == len(loft_verts)
    assert len(border_profile) - 1 == len(loft_verts) - 1 == len(loft_tri) / 2
