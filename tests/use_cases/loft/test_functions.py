import matplotlib.tri as tri
import numpy as np
import pytest

from cfdmod.use_cases.loft.functions import (
    find_borders,
    flatten_vertices_and_get_triangles_as_list_of_indexes,
    generate_loft_triangles,
    get_angle_between,
    remove_edges_of_internal_holes,
    remove_edges_oposite_to_loft_direction,
    remove_edges_too_aligned_with_projection_direction,
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
    (
        flattened_vertices, tri_index_matrix 
    ) = flatten_vertices_and_get_triangles_as_list_of_indexes(triangle_vertices)
    
    yield tri_index_matrix


def test_find_border(nx, ny, triangle_indices):
    border_edges = find_borders(triangles_vertices=triangle_indices)
    expected_edge_count = (nx + ny) * 2
    
    assert len(border_edges) == expected_edge_count
    

def test_angle_between():
    vec1 = np.array([1, 0, 0])
    vec2 = np.array([0, 1, 0])
    vec3 = np.array([1, 1, 0])
    vec4 = np.array([-1, -1, 0])

    assert get_angle_between(vec1, vec2) == 90
    assert get_angle_between(vec1, vec3) == 45
    assert get_angle_between(vec1, vec4) == 135
    assert get_angle_between(vec2, vec3) == 45


def test_loft_surface(triangle_vertices):
    projection_diretion = np.array([1, 0, 0])
    flattened_vertices, tri_index_matrix = flatten_vertices_and_get_triangles_as_list_of_indexes(
        triangle_vertices=triangle_vertices
    )
    border_edges = find_borders(triangles_vertices=tri_index_matrix)
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

    border_edges = remove_edges_oposite_to_loft_direction(
        vertices=flattened_vertices,
        edges=border_edges,
        mesh_center=center,
        projection_diretion=projection_diretion,
    )

    border_edges = remove_edges_too_aligned_with_projection_direction(
        vertices=flattened_vertices,
        edges=border_edges,
        projection_diretion=projection_diretion,
        angle_tolerance=45,
    )

    loft_tri, loft_normals = generate_loft_triangles(
        vertices=flattened_vertices,
        edges=border_edges,
        projection_diretion=projection_diretion,
        loft_length=100,
        loft_z_pos=1,
        mesh_center=center,
    )
