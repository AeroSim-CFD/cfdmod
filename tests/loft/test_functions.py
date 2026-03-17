import matplotlib.tri as tri
import numpy as np
import pytest

from cfdmod.loft.functions import (
    find_borders,
    flatten_vertices_and_get_triangles_as_list_of_indexes,
    generate_loft_triangles,
    remove_edges_of_internal_holes,
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


@pytest.fixture()
def triangle_indices(triangle_vertices):
    (flattened_vertices, tri_index_matrix) = flatten_vertices_and_get_triangles_as_list_of_indexes(
        triangle_vertices
    )

    yield tri_index_matrix


def test_find_border(nx, ny, triangle_indices):
    border_edges = find_borders(triangle_vertices=triangle_indices)
    expected_edge_count = (nx + ny) * 2

    assert len(border_edges) == expected_edge_count


def test_loft_surface(triangle_vertices):
    flattened_vertices, tri_index_matrix = flatten_vertices_and_get_triangles_as_list_of_indexes(
        triangle_vertices=triangle_vertices
    )
    border_edges = find_borders(triangle_vertices=tri_index_matrix)
    border_edges = remove_edges_of_internal_holes(
        vertices=flattened_vertices,
        edges=border_edges,
    )

    center = np.array(
        [
            (flattened_vertices[:, 0].max() + flattened_vertices[:, 0].min()) / 2,
            (flattened_vertices[:, 1].max() + flattened_vertices[:, 1].min()) / 2,
            (flattened_vertices[:, 2].max() + flattened_vertices[:, 2].min()) / 2,
        ]
    )

    loft_geom = generate_loft_triangles(
        vertices=flattened_vertices,
        edges=border_edges,
        loft_radius=20.0,
        loft_z_pos=0.0,
        mesh_center=center,
    )

    assert loft_geom is not None
    assert loft_geom.triangle_vertices.shape[1] == 3
    assert loft_geom.triangle_vertices.shape[2] == 3
