import math
import pathlib

import numpy as np
import pymeshlab
from pymeshlab import MeshSet, PureValue


def flatten_vertices_and_get_triangles_as_list_of_indexes(
    triangle_vertices: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Separates coordinates of vertices of faces from their indices.

    Args:
        triangle_vertices (np.ndarray[t,3,3]): values as coordinates of the 3 vertices
            coordinates:
                0 - number of triangle
                1 - number of vertice
                2 - coordinate of vertice

    Returns:
        tuple[np.ndarray, np.ndarray]: Flattened vertices and triangles specified as vertices indexes:
            - flattened_vertices (np.ndarray[v,3]): values as vertices coordinates
                coordinates:
                    0 - number of vertice
                    1 - vertice coordinate
            - triangles (np.ndarray[t,3]): values as triangles vertices ids
                coordinates:
                    0 - number of triangle
                    1 - number of vertice
    """

    def _get_float_as_int(v: float) -> int:
        return int(v * 10**decimals)

    def _get_as_key(v: np.ndarray) -> tuple[int]:
        return tuple(_get_float_as_int(vv) for vv in v)

    s = triangle_vertices.shape
    flattened_vertices = triangle_vertices.reshape((s[0] * s[1], 3))

    # Round for comparison
    decimals = 5
    flat_indexes = {_get_as_key(v): i for i, v in enumerate(flattened_vertices)}

    # Indexed as [t_idx, edge_idx] = (v0, v1)
    tri_index_matrix = np.empty((s[0], 3), dtype=np.uint32)

    for t_idx, tri in enumerate(triangle_vertices):
        v_idxs = []
        for v in tri:
            key = _get_as_key(v)
            val = flat_indexes[key]
            v_idxs.append(val)
        tri_index_matrix[t_idx] = v_idxs

    return flattened_vertices, tri_index_matrix


def find_borders(triangles_vertices: np.ndarray) -> np.ndarray:
    """Identify edges of border, based on repetition of edges

    Args:
        triangles_vertices (np.ndarray[t,3]): values as triangles vertices ids
            coordinates:
                0 - number of triangle
                1 - number of vertice

    Returns:
        edges (np.ndarray[e,2]): edges from border, indetified by two vertices indexes
            coordinates:
                0 - number of edge
                1 - number of edge vertice
    """
    n_triangles = len(triangles_vertices)
    triangles_edges = np.empty((n_triangles, 3, 2), dtype=np.uint32)
    for t_idx, tri in enumerate(triangles_vertices):
        triangles_edges[t_idx] = [
            tuple(sorted([tri[id_0], tri[id_1]])) for id_0, id_1 in [(0, 1), (0, 2), (1, 2)]
        ]
    flat_edges = triangles_edges.reshape(n_triangles * 3, 2)
    flat_edges_tp = [tuple(edge) for edge in flat_edges]

    unseen_edges = set(flat_edges_tp)
    unique_edges = set()
    for edge in flat_edges_tp:
        if edge in unseen_edges:
            unseen_edges.remove(edge)
            unique_edges.add(edge)
        elif edge in unique_edges:
            unique_edges.remove(edge)

    return np.array(list(unique_edges))


def remove_edges_of_internal_holes(vertices: np.ndarray, edges: np.ndarray) -> np.ndarray:
    """Remove border edges comming from internal holes

    Args:
        vertices (np.ndarray[v,3]): values as vertices coordinates
                coordinates:
                    0 - number of vertice
                    1 - vertice coordinate
        edges (np.ndarray[e,2]): edges from border, indetified by two vertices indexes
            coordinates:
                0 - number of edge
                1 - number of edge vertice

    Returns:
        np.ndarray[e,2]: filtered edges from border, indetified by two vertices indexes.
    """
    edges_by_vertex = [set() for v in range(0, vertices.shape[0])]
    vertices_to_analyze = set()
    for edge in edges:
        edges_by_vertex[edge[0]].add(tuple(edge))
        vertices_to_analyze.add(int(edge[0]))
        edges_by_vertex[edge[1]].add(tuple(edge))
        vertices_to_analyze.add(int(edge[1]))

    groups = []
    while len(vertices_to_analyze) > 0:
        group = {"vertices_id": set(), "edges": set()}
        current_vertice = vertices_to_analyze.pop()
        while len(edges_by_vertex[current_vertice]) > 0:
            current_edge = edges_by_vertex[current_vertice].pop()
            group["vertices_id"].add(current_vertice)
            group["edges"].add(current_edge)
            next_vertice = (
                current_edge[0] if current_edge[1] == current_vertice else current_edge[1]
            )
            edges_by_vertex[next_vertice].remove(current_edge)
            vertices_to_analyze.discard(next_vertice)
            current_vertice = next_vertice
        groups.append(group)

    # find group with biggest diameter
    groups_diameter = []
    for group in groups:
        vertices_id = list(group["vertices_id"])
        x = [vertices[v, 0] for v in vertices_id]
        y = [vertices[v, 1] for v in vertices_id]
        z = [vertices[v, 2] for v in vertices_id]
        x_ampl = max(x) - min(x)
        y_ampl = max(y) - min(y)
        z_ampl = max(z) - min(z)
        x_ampl = 1e-16 if x_ampl == 0 else x_ampl
        y_ampl = 1e-16 if y_ampl == 0 else y_ampl
        z_ampl = 1e-16 if z_ampl == 0 else z_ampl
        groups_diameter.append(x_ampl**2 + y_ampl**2 + z_ampl**2)

    max_diam = max(groups_diameter)
    max_diam_group_id = groups_diameter.index(max_diam)
    biggest_group = groups[max_diam_group_id]

    return np.array(list(biggest_group["edges"]))


def remove_edges_too_aligned_with_projection_direction(
    vertices: np.ndarray,
    edges: np.ndarray,
    projection_diretion: np.ndarray,
    angle_tolerance: float,
) -> np.ndarray:
    """Remove border edges that are too aligned with the projection direction.
    Faces created based on those edges would have very poor quality

    Args:
        vertices (np.ndarray[v,3]): values as vertices coordinates
                coordinates:
                    0 - number of vertice
                    1 - vertice coordinate
        edges (np.ndarray[e,2]): edges from border, indetified by two vertices indexes
            coordinates:
                0 - number of edge
                1 - number of edge vertice
        projection_diretion (np.ndarray[3]): vector with the direction where loft will be projected
        angle_tolerance (float): threshold of angle tolerance.
            - If angle between edge and projection_direction gets below this value, edge is removed from selection

    Returns:
        np.ndarray[e,2]: filtered edges from border, indetified by two vertices indexes.
    """
    edge_directions = vertices[edges[:, 1], :] - vertices[edges[:, 0], :]
    edge_directions[:, 2] = 0
    angles_between_edges_and_projection = get_angle_between(
        ref_vec=edge_directions, target_vec=projection_diretion
    )
    mask_not_too_aligned = (angle_tolerance < angles_between_edges_and_projection) & (
        angles_between_edges_and_projection < 180 - angle_tolerance
    )
    return edges[mask_not_too_aligned]


def _unit_vector(vector):
    """Returns the unit vector of the vector."""
    if vector.ndim > 1:
        return vector / np.linalg.norm(vector, axis=1, keepdims=True)
    else:
        return vector / np.linalg.norm(vector)


def get_angle_between(ref_vec: np.ndarray, target_vec: np.ndarray) -> float:
    """Returns the angle in radians between vectors 'ref_vec' and 'target_vec'

    Args:
        ref_vec (np.ndarray): Reference vector
        target_vec (np.ndarray): Target vector

    Returns:
        float: Angle between vectors 'ref_vec' and 'target_vec in degrees
    """
    ref_vec_u = _unit_vector(ref_vec)
    target_vec_u = _unit_vector(target_vec)
    angle_rad = np.arccos(np.dot(ref_vec_u, target_vec_u))

    return np.degrees(angle_rad)


def remove_edges_oposite_to_loft_direction(
    vertices: np.ndarray,
    edges: np.ndarray,
    projection_diretion: np.ndarray,
    mesh_center: np.ndarray,
) -> np.ndarray:
    """Remove border edges that are on the oposite side of the projection direction.

    Args:
        vertices (np.ndarray[v,3]): values as vertices coordinates
                coordinates:
                    0 - number of vertice
                    1 - vertice coordinate
        edges (np.ndarray[e,2]): edges from border, indetified by two vertices indexes
            coordinates:
                0 - number of edge
                1 - number of edge vertice
        projection_diretion (np.ndarray[3]): vector with the direction where loft will be projected
        mesh_center (np.ndarray[3]): Nominal center of mesh. Defines what 'oposite side' means

    Returns:
        np.ndarray[e,2]: filtered edges from border, indetified by two vertices indexes.
    """
    vertices_0 = vertices[edges[:, 0]]
    vertices_1 = vertices[edges[:, 1]]
    vert_dir_from_center_0 = vertices_0 - mesh_center
    vert_dir_from_center_1 = vertices_1 - mesh_center
    angles_0 = get_angle_between(ref_vec=vert_dir_from_center_0, target_vec=projection_diretion)
    angles_1 = get_angle_between(ref_vec=vert_dir_from_center_1, target_vec=projection_diretion)
    mask_vertices_on_right_side = (angles_0 < 90) & (angles_1 < 90)

    return edges[mask_vertices_on_right_side]


def generate_loft_triangles(
    vertices: np.ndarray,
    edges: np.ndarray,
    projection_diretion: np.ndarray,
    mesh_center: np.ndarray,
    loft_length: float,
    loft_z_pos: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Generates STL of loft

    Note that terrain and loft vertices are assumed to be ordered and aligned correctly

    Args:
        vertices (np.ndarray[v,3]): values as vertices coordinates
                coordinates:
                    0 - number of vertice
                    1 - vertice coordinate
        edges (np.ndarray[e,2]): edges from border, indetified by two vertices indexes
            coordinates:
                0 - number of edge
                1 - number of edge vertice
        projection_diretion (np.ndarray[3]): vector with the direction where loft will be projected
        mmesh_center (np.ndarray[3]): Nominal center of mesh. Used to contextualize loft length
        loft_length: float,
        loft_z_pos: float,

    Returns:
        tuple[np.ndarray, np.ndarray]: Tuple with loft surface triangles and normals
    """

    def _get_distance_from_center_in_the_projection_direction(
        vertices: np.ndarray, mesh_center: np.ndarray, projection_diretion: np.ndarray
    ) -> np.ndarray:
        vert_dir_from_center = vertices - mesh_center
        return np.dot(vert_dir_from_center, projection_diretion)

    def _normal_of_triangles(triangles: np.ndarray) -> np.ndarray:
        v0 = triangles[:, 0, :].squeeze()
        v1 = triangles[:, 1, :].squeeze()
        v2 = triangles[:, 2, :].squeeze()
        u = v1 - v0
        v = v2 - v0
        return np.cross(u, v)

    num_edges_on_border = edges.shape[0]
    vertices_on_boder_id = edges.reshape(num_edges_on_border * 2)
    vertices_on_border = vertices[vertices_on_boder_id, :]
    loft_distantce_start = np.max(
        _get_distance_from_center_in_the_projection_direction(
            vertices=vertices_on_border,
            mesh_center=mesh_center,
            projection_diretion=projection_diretion,
        )
    )
    loft_distantce_end = loft_distantce_start + loft_length

    edge_verts = []
    edge_verts.append(vertices[edges[:, 0]])
    edge_verts.append(vertices[edges[:, 1]])
    edge_verts_projection = []

    for vertices in edge_verts:
        distance_of_vertices_from_center = _get_distance_from_center_in_the_projection_direction(
            vertices=vertices,
            mesh_center=mesh_center,
            projection_diretion=projection_diretion,
        )
        distance_to_project = loft_distantce_end - distance_of_vertices_from_center

        projected_vertices = vertices + distance_to_project[:, np.newaxis] * projection_diretion
        projected_vertices[:, 2] = loft_z_pos
        edge_verts_projection.append(projected_vertices)

    triangles_0 = np.stack([edge_verts[0], edge_verts[1], edge_verts_projection[1]], axis=1)
    triangles_1 = np.stack(
        [edge_verts[0], edge_verts_projection[1], edge_verts_projection[0]], axis=1
    )

    full_triangles = np.concatenate([triangles_0, triangles_1], axis=0)
    full_normals = _normal_of_triangles(full_triangles)
    mask_inverted_nomals = (full_normals[:, 2]).squeeze() < 0
    corrected_triangles = full_triangles.copy()

    corrected_triangles[mask_inverted_nomals, 0, :] = full_triangles[mask_inverted_nomals, 1, :]
    corrected_triangles[mask_inverted_nomals, 1, :] = full_triangles[mask_inverted_nomals, 0, :]
    corrected_normals = _normal_of_triangles(corrected_triangles)

    return corrected_triangles, corrected_normals


def generate_loft_surface(
    triangle_vertices: np.ndarray,
    projection_diretion: np.ndarray,
    loft_length: float,
    loft_z_pos: float,
    angle_tolerance_for_alignement_edge_projection: float = 45,
) -> tuple[np.ndarray, np.ndarray]:
    """Generate loft surface (triangles and normals)

    Args:
        triangle_vertices (np.ndarray[t,3,3]): values as coordinates of the 3 vertices
            coordinates:
                0 - number of triangle
                1 - number of vertice
                2 - coordinate of vertice
        projection_diretion (np.ndarray[3]): vector with the direction where loft will be projected
        loft_length (float): Minimum length of loft
        loft_z_pos (float): Target z position
        angle_tolerance_for_alignement_edge_projection (float)(default=45): Maximum alignment tolerated to avoid triangles with bad quality

    Returns:
        tuple[np.ndarray, np.ndarray]: Loft triangles and normals
    """
    projection_diretion = _unit_vector(projection_diretion)
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
        angle_tolerance=angle_tolerance_for_alignement_edge_projection,
    )

    loft_tri, loft_normals = generate_loft_triangles(
        vertices=flattened_vertices,
        edges=border_edges,
        projection_diretion=projection_diretion,
        loft_length=loft_length,
        loft_z_pos=loft_z_pos,
        mesh_center=center,
    )

    return loft_tri, loft_normals


# def apply_remeshing(
#     element_size: float,
#     mesh_path: pathlib.Path,
#     output_path: pathlib.Path,
#     crease_angle: float = 89,
# ):
#     """Create a remeshed surface from input mesh

#     Args:
#         element_size (float): Target element size
#         mesh_path (pathlib.Path): Original mesh path
#         output_path (pathlib.Path): Output mesh path
#         crease_angle (float): Minimal angle for preserving edges
#     """
#     ms: MeshSet = pymeshlab.MeshSet()
#     ms.load_new_mesh(str(mesh_path.absolute()))
#     ms.meshing_isotropic_explicit_remeshing(
#         iterations=15, targetlen=PureValue(element_size), featuredeg=crease_angle
#     )
#     ms.compute_selection_by_condition_per_face(condselect="fnz<0")
#     ms.meshing_invert_face_orientation(onlyselected=True)
#     ms.save_current_mesh(str(output_path.absolute()), binary=True)


# def rotate_vector_around_z(vector: np.ndarray, angle_degrees: float) -> np.ndarray:
#     """Rotates a vector around z axis from a given angle

#     Args:
#         vector (np.ndarray): Vector to be rotated (x, y, z)
#         angle_degrees (float): Angle of rotation in degrees

#     Returns:
#         np.ndarray: Rotated 3D vector
#     """
#     angle_radians = np.radians(angle_degrees)
#     rotation_matrix = np.array(
#         [
#             [np.cos(angle_radians), -np.sin(angle_radians), 0],
#             [np.sin(angle_radians), np.cos(angle_radians), 0],
#             [0, 0, 1],
#         ]
#     )
#     rotated_vector = np.dot(rotation_matrix, vector)

#     return rotated_vector
