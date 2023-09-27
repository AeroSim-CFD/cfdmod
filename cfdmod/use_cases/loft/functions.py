import math
from typing import Tuple

import numpy as np


def find_border(vertices: np.ndarray) -> set:
    """Extract vertices and unique edges for the stl file's border in x-y plane

    Args:
        stl_file (STLFile): STL file containing terrain mesh.

    Returns:
        Tuple[np.ndarray, set]: Tuple containing the mesh's vertices and its border edges vertices
    """
    s = vertices.shape

    flattened_vertices = vertices.reshape((s[0] * s[1], 3))

    # Round for comparison
    decimals = 0

    get_float_as_int = lambda v: int(v * 10**decimals)
    get_as_key = lambda v: tuple(get_float_as_int(vv) for vv in v)

    flat_indexes = {get_as_key(v): i for i, v in enumerate(flattened_vertices)}

    # Indexed as [t_idx, edge_idx] = (v0, v1)
    tri_index_matrix = np.empty((s[0], 3, 2), dtype=np.uint32)

    for t_idx, tri in enumerate(vertices):
        v_idxs = []
        for v in (tri.v0, tri.v1, tri.v2):
            key = get_as_key(v)
            val = flat_indexes[key]
            v_idxs.append(val)
        tri_edges = [tuple(sorted((v_idxs[i], v_idxs[j]))) for i, j in [(0, 1), (1, 2), (2, 0)]]
        tri_index_matrix[t_idx] = tri_edges

    n_triangles = len(triangles)
    flat_edges = tri_index_matrix.reshape(n_triangles * 3, 2)
    flat_edges_tp = [tuple(edge) for edge in flat_edges]

    unseen_edges = set(flat_edges_tp)
    unique_edges = set()
    for edge in flat_edges_tp:
        if edge in unseen_edges:
            unseen_edges.remove(edge)
            unique_edges.add(edge)
        elif edge in unique_edges:
            unique_edges.remove(edge)

    return flattened_vertices, unique_edges


def project_border(
    unique_edges: set,
    flattened_vertices: np.ndarray,
    flow_angle: int,
    center: Tuple[float, float, float],
    size: Tuple[float, float],
):
    """_summary_

    Args:
        unique_edges (set): _description_
        flattened_vertices (np.ndarray): _description_
        flow_angle (int): _description_
        center (Tuple[float, float, float]): _description_
        size (Tuple[float, float]): _description_
    """

    def get_vector_from_points(
        p0: Tuple[float, float, float], p1: Tuple[float, float, float]
    ) -> Tuple[float, float]:
        """Return a tuple with a vector from p0 to p1 in the x-y plane"""
        return (p1[0] - p0[0], p1[1] - p0[1])

    def get_angle_between(ref_vec, target_vec):
        """Returns the angle in radians between vectors 'ref_vec' and 'target_vec'"""

        def unit_vector(vector):
            """Returns the unit vector of the vector."""
            return vector / np.linalg.norm(vector)

        ref_vec_u = unit_vector(ref_vec)
        target_vec_u = unit_vector(target_vec)

        angle = np.rad2deg(np.arccos(np.dot(ref_vec_u, target_vec_u)))
        dot = (
            ref_vec_u[0] * target_vec_u[0] + ref_vec_u[1] * target_vec_u[1]
        )  # Dot product between [x1, y1] and [x2, y2]
        det = ref_vec_u[0] * target_vec_u[1] - ref_vec_u[1] * target_vec_u[0]  # Determinant
        angle = -np.rad2deg(np.arctan2(det, dot))  # atan2(y, x) or atan2(sin, cos)

        return angle + 360 if angle < 0 else angle

    projection_angle = math.radians(flow_angle)

    border_indexes = [point_index for edge in unique_edges for point_index in edge]

    p0 = (
        center[0] - size[0] * math.sin(projection_angle) / 2,
        center[1] + size[1] * math.cos(projection_angle) / 2,
    )
    p1 = (
        center[0] + size[0] * math.sin(projection_angle) / 2,
        center[1] - size[1] * math.cos(projection_angle) / 2,
    )

    separation_line = get_vector_from_points(p0, p1)

    profile_vertices = np.empty((0, 3))

    for p_index in np.unique(border_indexes):
        target_point = flattened_vertices[p_index]
        if np.cross(separation_line, get_vector_from_points(p1, target_point)) >= 0:
            profile_vertices = np.vstack((profile_vertices, target_point))

    theta_sort = lambda x: get_angle_between(
        ref_vec=get_vector_from_points(center, p0), target_vec=get_vector_from_points(center, x)
    )

    profile_vertices = np.array(sorted(profile_vertices, key=theta_sort))

    return profile_vertices
