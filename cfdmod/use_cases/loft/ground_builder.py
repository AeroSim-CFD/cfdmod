from __future__ import annotations

# import math
import pathlib

import numpy as np
# import pymeshlab
# import trimesh
from cfdmod.api.geometry.STL import export_stl, read_stl

# from cfdmod.use_cases.loft.functions import generate_loft_surface
# from cfdmod.use_cases.roughness_gen import build_single_element, linear_pattern
# from cfdmod.use_cases.roughness_gen import parameters as cfdmod_parameters
# from lnas import LnasFormat
# from pymeshlab import MeshSet, PureValue

# from aero.models import Body, BodyType, Box
# from aero.units import (
#     Angle,
#     CoordinatesOperator,
#     CoordinateSystem,
#     CoordinateSystemName,
#     Quantity,
#     UnitName,
#     UnitsOperator,
# )
# from aero.utils import apply_sequence_of_transformations, generate_lnas_from_stl


class GroundBuilder:
    @classmethod
    def generate_loft_stl(
        cls,
        loft: Body,
        units_operator: UnitsOperator,
        coordinates_operator: CoordinatesOperator,
        stls_root_path: pathlib.Path,
        terrain: Body,
    ):
        loft_angle_rad = loft.properties.geometrical_angle.radians
        loft_primary_directions = {
            "upwind": np.array([np.cos(loft_angle_rad), np.sin(loft_angle_rad), 0]),
            "downwind": -np.array([np.cos(loft_angle_rad), np.sin(loft_angle_rad), 0]),
        }
        loft_secondary_directions = {
            "left": np.array([-np.sin(loft_angle_rad), np.cos(loft_angle_rad), 0]),
            "right": -np.array([-np.sin(loft_angle_rad), np.cos(loft_angle_rad), 0]),
        }

        loft_length_mag = {}
        for d in loft.properties.loft_length:
            loft_length_mag[d] = units_operator.magnitude(
                loft.properties.loft_length[d], unit_to=loft.length_unit_name
            )
        loft_tri = {}
        loft_normals = {}
        terrain_triangles, _ = read_stl(
            stls_root_path / terrain.file_relative_path.with_suffix(".stl")
        )

        radius_mag = units_operator.magnitude(terrain.properties.R, unit_to=loft.length_unit_name)
        if loft.properties.upwind_elevation.coordinate_system_name != loft.coordinate_system_name:
            upwind_elevation_coord = coordinates_operator.transform_to(
                loft.properties.upwind_elevation,
                new_coordinate_system_name=loft.coordinate_system_name,
            )
        else:
            upwind_elevation_coord = loft.properties.upwind_elevation
        upwind_elevation_mag = units_operator.magnitude(
            upwind_elevation_coord.position[2], unit_to=loft.length_unit_name
        )
        for side in loft_primary_directions:
            loft_tri[side], loft_normals[side] = generate_loft_surface(
                triangle_vertices=terrain_triangles,
                projection_diretion=loft_primary_directions[side],
                loft_length=loft_length_mag[side],
                loft_z_pos=upwind_elevation_mag,
                filter_radius=radius_mag * 0.99,
                keep_only_one_part=True,
            )

        primary_ground_triangles = np.concatenate(
            [loft_tri[side] for side in loft_tri] + [terrain_triangles]
        )
        for side in loft_secondary_directions:
            loft_tri[side], loft_normals[side] = generate_loft_surface(
                triangle_vertices=primary_ground_triangles,
                projection_diretion=loft_secondary_directions[side],
                loft_length=loft_length_mag[side],
                loft_z_pos=upwind_elevation_mag,
                filter_radius=radius_mag * 0.99,
                keep_only_one_part=False,
                tolerance=1,
            )
        full_loft_triangles = np.concatenate([loft_tri[side] for side in loft_tri])
        full_loft_normals = np.concatenate([loft_normals[side] for side in loft_normals])

        loft_relative_path = loft.file_relative_path
        loft_file_path = stls_root_path / loft_relative_path.with_suffix(".stl")
        export_stl(loft_file_path, full_loft_triangles, full_loft_normals)

        terrain_refinement = units_operator.magnitude(
            terrain.properties.refinement, unit_to=terrain.length_unit_name
        )

        # if loft.properties.remesh_loft:
        #     cls.__apply_remeshing_with_radius(
        #         element_size_min=terrain_refinement,
        #         radius_min=radius_mag,
        #         mesh_path=loft_file_path,
        #         output_path=loft_file_path,
        #     )

    # @classmethod
    # def __apply_remeshing_with_radius(
    #     self,
    #     element_size_min: float,
    #     radius_min: float,
    #     mesh_path: pathlib.Path,
    #     output_path: pathlib.Path,
    #     element_size_max: float = 100,
    #     offset: float = 200,
    # ):
    #     ms: MeshSet = pymeshlab.MeshSet()
    #     ms.load_new_mesh(str(mesh_path.absolute()))
    #     target_length = element_size_min
    #     max_length = element_size_max
    #     ms.meshing_isotropic_explicit_remeshing(
    #         iterations=10, targetlen=PureValue(75), featuredeg=30, selectedonly=False, adaptive=True,
    #     )

    #     for i in range(0):
    #         r_min = radius_min + (offset * i)
    #         r_min_filter = f"(( ((x0+x1+x2)/3)^2 + ((y0+y1+y2)/3)^2)^0.5 >= {r_min})"
    #         if target_length > max_length:
    #             ms.compute_selection_by_condition_per_face(condselect=r_min_filter)
    #             ms.apply_selection_dilatation()
    #             ms.meshing_isotropic_explicit_remeshing(
    #                 iterations=10,
    #                 targetlen=PureValue(target_length),
    #                 featuredeg=180,
    #                 selectedonly=True,
    #             )
    #             break
    #         r_max = radius_min + (offset * (i + 1))
    #         r_max_filter = f"(( ((x0+x1+x2)/3)^2 + ((y0+y1+y2)/3)^2)^0.5 <= {r_max})"
    #         filter = f"{r_min_filter} && {r_max_filter}"
    #         if target_length == 1:
    #             ms.compute_selection_by_condition_per_face(condselect=r_max_filter)
    #             ms.apply_selection_dilatation()
    #             ms.meshing_isotropic_explicit_remeshing(
    #                 iterations=10,
    #                 targetlen=PureValue(target_length),
    #                 featuredeg=180,
    #                 selectedonly=True,
    #             )
    #             continue
    #         ms.compute_selection_by_condition_per_face(condselect=filter)
    #         ms.apply_selection_dilatation()
    #         ms.meshing_isotropic_explicit_remeshing(
    #             iterations=10,
    #             targetlen=PureValue(target_length),
    #             featuredeg=180,
    #             selectedonly=True,
    #         )
    #         target_length = target_length * 1.2
    #     ms.meshing_remove_null_faces()
    #     ms.meshing_merge_close_vertices(
    #         threshold=pymeshlab.PercentageValue(0.02)  # very small threshold
    #     )
    #     ms.save_current_mesh(str(output_path.absolute()), binary=True)


def flatten_vertices_and_get_triangles_as_list_of_indexes(
    triangle_vertices: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Takes a set of faces defined as 3 vertices with coordinates and gives back the vertices separated in and the
    triangles specified by the vertices indexes

    Args:
        triangle_vertices[a,b,c] (np.ndarray): 3d numpy array. Coordinates ->
            0 - number of triangle (len = number of triangles)
            1 - number of vertice (len = 3)
            2 - coordinate of verice (len = 3)

    Returns:
        tuple[np.ndarray, np.ndarray]: Flattened vertices and triangles specified as vertices indexes:
            - flattened_vertices[d,c] -> number of vertices, vertice coordinate
            - triangles[a,c] -> number of triangle, number of triangle vertice (3)
    """

    def _get_float_as_int(v: float) -> int:
        return int(v * 10**decimals)

    def _get_as_key(v: np.ndarray) -> tuple[int]:
        return tuple(_get_float_as_int(vv) for vv in v)

    s = triangle_vertices.shape
    flattened_vertices = triangle_vertices.reshape((s[0] * s[1], 3))

    # Round for comparison
    decimals = 10

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


def find_borders(triangle_vertices: np.ndarray) -> np.ndarray:
    """Identify edges of border, based on repetition of edges

    Args:
        triangle_vertices (np.ndarray): vertices indexes

    Returns:
        set[tuple[int,int]]: selected set of edges, indetified by vertices indexes
    """
    n_triangles = len(triangle_vertices)
    triangles_edges = np.empty((n_triangles, 3, 2), dtype=np.uint32)
    for t_idx, tri in enumerate(triangle_vertices):
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
    """Remove border vertices comming from internal holes

    Args:
        border_verts (np.ndarray): All border vertices
        radius (float): Internal radius where to ignore edges

    Returns:
        tuple[np.ndarray, np.ndarray]: Updated filtered border vertices and border edges
        that are not from internal holes
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
        groups_diameter.append(x_ampl**2 + y_ampl**2 + z_ampl**2)
    max_diam = max(groups_diameter)
    max_diam_group_id = groups_diameter.index(max_diam)
    biggest_group = groups[max_diam_group_id]

    return np.array(list(biggest_group["edges"]))


def remove_edges_too_aligned_with_projection_direction(
    vertices: np.ndarray,
    edges: np.ndarray,
    projection_diretion: np.ndarray,
    tolerance: float,
) -> np.ndarray:
    edge_directions = vertices[edges[:, 1], :] - vertices[edges[:, 0], :]
    edge_directions[:, 2] = 0
    angles_between_edges_and_projection = get_angle_between(
        ref_vec=edge_directions, target_vec=projection_diretion
    )
    mask_not_too_aligned = (tolerance < angles_between_edges_and_projection) & (
        angles_between_edges_and_projection < 180 - tolerance
    )
    return edges[mask_not_too_aligned]


def unit_vector(vector):
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
    ref_vec_u = unit_vector(ref_vec)
    target_vec_u = unit_vector(target_vec)
    angle_rad = np.arccos(np.dot(ref_vec_u, target_vec_u))

    return np.degrees(angle_rad)


def remove_edges_oposite_to_loft_direction(
    vertices: np.ndarray,
    edges: np.ndarray,
    projection_diretion: np.ndarray,
    mesh_center: np.ndarray,
) -> np.ndarray:

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
        border_profile (np.ndarray): Vertices in terrain
        loft_vertices (np.ndarray): Vertices generated for loft

    Returns:
        tuple[np.ndarray, np.ndarray]: Tuple with loft surface triangles and normals
    """

    def get_distance_from_center_in_the_projection_direction(
        vertices: np.ndarray, mesh_center: np.ndarray, projection_diretion: np.ndarray
    ) -> np.ndarray:
        vert_dir_from_center = vertices - mesh_center
        return np.dot(vert_dir_from_center, projection_diretion)

    def normal_of_triangles(triangles: np.ndarray) -> np.ndarray:
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
        get_distance_from_center_in_the_projection_direction(
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
        distance_of_vertices_from_center = get_distance_from_center_in_the_projection_direction(
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
    full_normals = normal_of_triangles(full_triangles)
    mask_inverted_nomals = (full_normals[:, 2]).squeeze() < 0
    corrected_triangles = full_triangles.copy()
    corrected_triangles[mask_inverted_nomals, 0, :] = full_triangles[mask_inverted_nomals, 1, :]
    corrected_triangles[mask_inverted_nomals, 1, :] = full_triangles[mask_inverted_nomals, 0, :]
    corrected_normals = normal_of_triangles(corrected_triangles)
    return corrected_triangles, corrected_normals


def generate_loft_surface(
    triangle_vertices: np.ndarray,
    projection_diretion: np.ndarray,
    loft_length: float,
    loft_z_pos: float,
    filter_radius: float,
    keep_only_one_part: bool,
    tolerance: float = 30,
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
    projection_diretion = unit_vector(projection_diretion)

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
        tolerance=tolerance,
    )

    # if keep_only_one_part:
    #     border_edges = remove_edges_of_internal_holes(
    #         vertices = flattened_vertices,
    #         edges = border_edges,
    #     )

    loft_tri, loft_normals = generate_loft_triangles(
        vertices=flattened_vertices,
        edges=border_edges,
        projection_diretion=projection_diretion,
        loft_length=loft_length,
        loft_z_pos=loft_z_pos,
        mesh_center=center,
    )

    return loft_tri, loft_normals
