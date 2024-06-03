import math
import pathlib

import numpy as np
import pymeshlab
from pymeshlab import AbsoluteValue, MeshSet


def find_border(triangle_vertices: np.ndarray) -> np.ndarray:
    """Extract vertices from unique edges from stl file's border with its triangles and normals

    Args:
        triangle_vertices (np.ndarray): STL triangles

    Returns:
        np.ndarray: Vertices from the border
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

    return flattened_vertices[np.unique(border_indexes)]


def remove_vertices_from_internal_holes(border_verts: np.ndarray, radius: float) -> np.ndarray:
    """Remove border vertices comming from internal holes

    Args:
        border_verts (np.ndarray): All border vertices
        radius (float): Internal radius where to ignore edges

    Returns:
        tuple[np.ndarray, np.ndarray]: Updated filtered border vertices and border edges
        that are not from internal holes
    """
    mask = np.where(np.sum((border_verts**2), axis=1) ** 0.5 > radius)[0]

    return border_verts[mask]


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


def project_border(
    border_vertices: np.ndarray, projection_diretion: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Projects the border vertices based on wind source direction

    Args:
        border_vertices (np.ndarray): Border vertices
        projection_diretion (np.ndarray): Direction which to project the border

    Returns:
        tuple[np.ndarray, np.ndarray]: Ordered border profile vertices and mesh center coordinate
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

    return profile_vertices, center


def generate_circular_loft_vertices(
    border_profile: np.ndarray,
    projection_diretion: np.ndarray,
    loft_length: float,
    loft_z_pos: float,
    mesh_center: np.ndarray,
) -> np.ndarray:
    """Generates vertices to use for building loft

    Args:
        border_profile (np.ndarray): Vertices in terrain that will connect to loft
        projection_diretion (np.ndarray): Direction which to project the border
        loft_length (float): Loft's length
        loft_z_pos (float): Loft's position in z (height)
        mesh_center (np.ndarray): Coordinate for the center of the surface mesh.

    Returns:
        np.ndarray: Vertices to use for building loft, aligned and ordered with border profile
    """

    loft_vertices: list[np.ndarray] = []

    normal = projection_diretion
    normal[2] = 0

    max_center_distance = np.array([np.dot(x - mesh_center, normal) for x in border_profile]).max()

    for v in border_profile:
        v_center_distance = np.dot(v - np.array(mesh_center), normal)
        sum_arr = np.array(
            [
                (loft_length + (max_center_distance - v_center_distance)) * normal[0],
                (loft_length + (max_center_distance - v_center_distance)) * normal[1],
                0,
            ],
            dtype=np.float32,
        )
        # Move vertice to border
        loft_v = v + sum_arr
        # Update height of vertice
        loft_v[2] = loft_z_pos
        loft_vertices.append(loft_v)

    return np.array(loft_vertices, dtype=np.float32)


def generate_loft_triangles(
    border_profile: np.ndarray, loft_vertices: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Generates STL of loft

    Note that terrain and loft vertices are assumed to be ordered and aligned correctly

    Args:
        border_profile (np.ndarray): Vertices in terrain
        loft_vertices (np.ndarray): Vertices generated for loft

    Returns:
        tuple[np.ndarray, np.ndarray]: Tuple with loft surface triangles and normals
    """

    def normal_from_vertices(v0: np.ndarray, v1: np.ndarray, v2: np.ndarray) -> np.ndarray:
        u = v1 - v0
        v = v2 - v0
        return np.cross(u, v)

    def fix_vertices_order(
        v0: np.ndarray, v1: np.ndarray, v2: np.ndarray
    ) -> tuple[list[np.ndarray], np.ndarray]:
        v = [v0, v1, v2]
        n = normal_from_vertices(*v)
        # Normal pointing to z negative, wrong order
        if n[2] < 0:
            # Change vertices order
            v = [v0, v2, v1]
            n = normal_from_vertices(*v)
        return v, n

    normals = []
    triangles = []
    for i in range(1, len(border_profile)):
        vt0, vt1 = border_profile[i - 1 : i + 1]
        vl0, vl1 = loft_vertices[i - 1 : i + 1]

        t0_v = [vt0, vl0, vt1]
        t1_v = [vl0, vl1, vt1]

        t0_v, t0_n = fix_vertices_order(*t0_v)
        t1_v, t1_n = fix_vertices_order(*t1_v)

        normals.append(t0_n)
        normals.append(t1_n)
        triangles.append(t0_v)
        triangles.append(t1_v)

    return np.array(triangles), np.array(normals)


def generate_loft_surface(
    triangle_vertices: np.ndarray,
    projection_diretion: np.ndarray,
    loft_length: float,
    loft_z_pos: float,
    filter_radius: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Generate loft surface (triangles and normals)

    Args:
        triangle_vertices (np.ndarray): Mesh triangle vertices
        projection_diretion (np.ndarray): Direction of loft projection
        loft_length (float): Minimum length of loft
        loft_z_pos (float): Target z position
        filter_radius (float): Internal filter radius to ignore edges

    Returns:
        tuple[np.ndarray, np.ndarray]: Loft triangles and normals
    """
    unfiltered_border_verts = find_border(triangle_vertices=triangle_vertices)
    border_verts = remove_vertices_from_internal_holes(
        border_verts=unfiltered_border_verts,
        radius=filter_radius,
    )
    border_profile, mesh_center = project_border(
        border_verts, projection_diretion=projection_diretion
    )
    loft_verts = generate_circular_loft_vertices(
        border_profile=border_profile,
        projection_diretion=projection_diretion,
        loft_length=loft_length,
        loft_z_pos=loft_z_pos,
        mesh_center=mesh_center,
    )
    loft_tri, loft_normals = generate_loft_triangles(
        border_profile=border_profile, loft_vertices=loft_verts
    )

    return loft_tri, loft_normals


def apply_remeshing(
    element_size: float,
    mesh_path: pathlib.Path,
    output_path: pathlib.Path,
    crease_angle: float = 89,
):
    """Create a remeshed surface from input mesh

    Args:
        element_size (float): Target element size
        mesh_path (pathlib.Path): Original mesh path
        output_path (pathlib.Path): Output mesh path
        crease_angle (float): Minimal angle for preserving edges
    """
    ms: MeshSet = pymeshlab.MeshSet()
    ms.load_new_mesh(str(mesh_path.absolute()))
    ms.meshing_isotropic_explicit_remeshing(
        iterations=15, targetlen=AbsoluteValue(element_size), featuredeg=crease_angle
    )
    ms.compute_selection_by_condition_per_face(condselect="fnz<0")
    ms.meshing_invert_face_orientation(onlyselected=True)
    ms.save_current_mesh(str(output_path.absolute()), binary=True)


def rotate_vector_around_z(vector: np.ndarray, angle_degrees: float) -> np.ndarray:
    """Rotates a vector around z axis from a given angle

    Args:
        vector (np.ndarray): Vector to be rotated (x, y, z)
        angle_degrees (float): Angle of rotation in degrees

    Returns:
        np.ndarray: Rotated 3D vector
    """
    angle_radians = np.radians(angle_degrees)
    rotation_matrix = np.array(
        [
            [np.cos(angle_radians), -np.sin(angle_radians), 0],
            [np.sin(angle_radians), np.cos(angle_radians), 0],
            [0, 0, 1],
        ]
    )
    rotated_vector = np.dot(rotation_matrix, vector)

    return rotated_vector
