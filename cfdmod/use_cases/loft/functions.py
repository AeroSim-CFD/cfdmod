import math

import numpy as np


def find_border(triangle_vertices: np.ndarray) -> tuple[np.ndarray, set]:
    """Extract vertices from unique edges from stl file's border with its triangles and normals

    Args:
        triangle_vertices (np.ndarray): STL triangles

    Returns:
        tuple[np.ndarray, set]: Vertices from the border and a set with border edges
    """
    s = triangle_vertices.shape

    flattened_vertices = triangle_vertices.reshape((s[0] * s[1], 3))

    # Round for comparison
    decimals = 0

    get_float_as_int = lambda v: int(v * 10**decimals)
    get_as_key = lambda v: tuple(get_float_as_int(vv) for vv in v)

    flat_indexes = {get_as_key(v): i for i, v in enumerate(flattened_vertices)}

    # Indexed as [t_idx, edge_idx] = (v0, v1)
    tri_index_matrix = np.empty((s[0], 3, 2), dtype=np.uint32)

    for t_idx, tri in enumerate(triangle_vertices):
        v_idxs = []
        for v in tri:
            key = get_as_key(v)
            val = flat_indexes[key]
            v_idxs.append(val)
        tri_edges = [tuple(sorted((v_idxs[i], v_idxs[j]))) for i, j in [(0, 1), (1, 2), (2, 0)]]
        tri_index_matrix[t_idx] = tri_edges

    n_triangles = len(triangle_vertices)
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
    border_indexes = [point_index for edge in unique_edges for point_index in edge]

    return flattened_vertices[np.unique(border_indexes)], unique_edges


def get_angle_between(ref_vec: np.ndarray, target_vec: np.ndarray) -> float:
    """Returns the angle in radians between vectors 'ref_vec' and 'target_vec'

    Args:
        ref_vec (np.ndarray): Reference vector
        target_vec (np.ndarray): Target vector

    Returns:
        float: Angle between vectors 'ref_vec' and 'target_vec in degrees
    """

    def unit_vector(vector):
        """Returns the unit vector of the vector."""
        return vector / np.linalg.norm(vector)

    ref_vec_u = unit_vector(ref_vec)
    target_vec_u = unit_vector(target_vec)
    cross_prod = np.cross(ref_vec_u, target_vec_u)

    angle_rad = np.arctan2(np.linalg.norm(cross_prod), np.dot(ref_vec_u, target_vec_u))
    angle = np.degrees(angle_rad)
    if cross_prod[2] < 0:
        angle = 360 - angle

    return angle + 360 if angle < 0 else angle


def project_border(border_vertices: np.ndarray, projection_diretion: np.ndarray) -> np.ndarray:
    """Projects the border vertices based on wind source direction

    Args:
        border_vertices (np.ndarray): Border vertices
        projection_diretion (np.ndarray): Direction which to project the border

    Returns:
        np.ndarray: Ordered border profile vertices
    """
    flow_angle = get_angle_between(
        ref_vec=(1, 0, 0),
        target_vec=projection_diretion,
    )
    projection_angle = math.radians(flow_angle)
    size = (
        border_vertices[:, 0].max() - border_vertices[:, 0].min(),
        border_vertices[:, 1].max() - border_vertices[:, 1].min(),
    )
    center = np.array(
        [
            (border_vertices[:, 0].max() + border_vertices[:, 0].min()) / 2,
            (border_vertices[:, 1].max() + border_vertices[:, 1].min()) / 2,
            (border_vertices[:, 2].max() + border_vertices[:, 2].min()) / 2,
        ]
    )

    p0 = np.array(
        [
            center[0] - size[0] * math.sin(projection_angle) / 2,
            center[1] + size[1] * math.cos(projection_angle) / 2,
            (border_vertices[:, 2].max() + border_vertices[:, 2].min()) / 2,
        ]
    )
    p1 = np.array(
        [
            center[0] + size[0] * math.sin(projection_angle) / 2,
            center[1] - size[1] * math.cos(projection_angle) / 2,
            (border_vertices[:, 2].max() + border_vertices[:, 2].min()) / 2,
        ]
    )

    separation_line = p1 - p0

    profile_vertices = np.empty((0, 3))

    for target_point in border_vertices:
        target_vec = target_point - p1
        if np.cross(separation_line[:2], target_vec[:2]) >= 0:
            profile_vertices = np.vstack((profile_vertices, target_point))

    theta_sort = lambda x: get_angle_between(ref_vec=p0 - center, target_vec=x - center)

    profile_vertices = np.array(sorted(profile_vertices, key=theta_sort))

    return profile_vertices
