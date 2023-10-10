import math
from itertools import product
from typing import List, Tuple

import numpy as np
import pymeshlab
from nassu.lnas import LagrangianGeometry
from scipy.spatial import ConvexHull, Delaunay

from cfdmod.use_cases.pressure.shape.intercept_points import PointsInterceptor
from cfdmod.use_cases.pressure.shape.meshing_functions import (
    find_intersection_point,
    is_in_range,
    line_plane_intersection,
    project_2d,
    remove_colinear,
    remove_duplicates_3d_points,
)
from cfdmod.use_cases.pressure.shape.regions import ZoningModel


def get_mesh_contour_polyline(
    mesh: LagrangianGeometry,
) -> Tuple[np.ndarray, Tuple[np.ndarray, np.ndarray]]:
    """Extracts the polyline that forms the contour of the surface represented by the mesh

    Args:
        mesh (LagrangianGeometry): Input LNAS Mesh object containing the mesh points and triangles

    Returns:
        Tuple[np.ndarray, Tuple[float, float]]: Tuple containing an array of the polyline vertices and a tuple with the mesh bounding box minimum and maximum
    """
    ms = pymeshlab.MeshSet()
    m = pymeshlab.Mesh(mesh.vertices, mesh.triangles)
    ms.add_mesh(m, "input_mesh")

    m = ms.current_mesh()
    bounding_box = m.bounding_box()
    bb_max = bounding_box.max()
    bb_min = bounding_box.min()
    ms.compute_selection_by_condition_per_face(condselect="(fnx>=0) || (fnx<=0)")
    ms.apply_filter("generate_polyline_from_selection_perimeter")
    m = ms.current_mesh()
    polyline_vertices = m.vertex_matrix()
    polyline_edges = m.edge_matrix()

    return (np.array(polyline_vertices), np.array(polyline_edges)), (
        np.array(bb_min),
        np.array(bb_max),
    )


def generate_polyline_edges(ordered_outline_vertices: np.ndarray) -> List[Tuple[int, int]]:
    """Generate tuples of polyline edges, conecting the vertices

    Args:
        ordered_outline_vertices (np.ndarray): Ordered polyline vertices

    Returns:
        List[Tuple[int, int]]: List of polyline edges, with tuples that connect points through their indices
    """
    primitive_poly_edges = []
    for i in range(len(ordered_outline_vertices)):
        if i == 0:
            primitive_poly_edges.append((len(ordered_outline_vertices) - 1, i))
        else:
            primitive_poly_edges.append((i - 1, i))

    return primitive_poly_edges


def clean_order_contour_verts(
    mesh_points: np.ndarray, mesh_triangles: np.ndarray, polyline_vertices: np.ndarray
) -> Tuple[np.ndarray, int]:
    """Clean any align vertices and order them according to normals

    Args:
        mesh_points (np.ndarray): Surface mesh points
        mesh_triangles (np.ndarray): Surface mesh triangles
        polyline_vertices (np.ndarray): Surface contour polyline vertices

    Returns:
        Tuple[np.ndarray, int]: Tuple containing the outline ordered vertices and the index of the normal, (x_aligned=1, y_aligned=2, z_aligned=3, not aligned=0)
    """
    # Assuming you have a normal vector for the polygon
    sample_triangle = np.random.choice(range(len(mesh_triangles) - 1), 1)[0]
    sample_plane_points = mesh_triangles[sample_triangle]
    p1 = mesh_points[sample_plane_points[0]]
    p2 = mesh_points[sample_plane_points[1]]
    p3 = mesh_points[sample_plane_points[2]]
    v1 = p2 - p1
    v2 = p3 - p2

    cross_product = np.cross(v1, v2)
    plane_normal = cross_product / np.linalg.norm(cross_product)

    # point_normalized = lambda point: point / np.linalg.norm(point)
    # dot_product = lambda point: np.dot(point_normalized(point), plane_normal)
    # dot_product_in_range = lambda point: max(min(dot_product(point), 1.0), -1.0)
    # angle = lambda point: np.arccos(dot_product_in_range(point))
    # angles = [angle(point) for point in polyline_vertices]
    # sorted_points = np.array([point for _, point in sorted(zip(angles, polyline_vertices))])

    # Project points to 2D
    # projected_points = project_2d(polyline_vertices, plane_normal)

    # Calculate the Convex Hull which gives us the vertices in counterclockwise order
    normal_index = 0

    if math.isclose(abs(plane_normal[0]), 1, rel_tol=0.1):
        # hull = ConvexHull(projected_points[:, 1:])
        normal_index = 1
    elif math.isclose(abs(plane_normal[1]), 1, rel_tol=0.1):
        # hull = ConvexHull(projected_points[:, ::2])
        normal_index = 2
    elif math.isclose(abs(plane_normal[2]), 1, rel_tol=0.1):
        # hull = ConvexHull(projected_points[:, :2])
        normal_index = 3
    # else:
    #     hull = ConvexHull(projected_points[:, 1:])

    # Get the ordered vertices
    # ordered_vertices = polyline_vertices[hull.vertices]

    # Remove colinear points
    # ordered_outline_vertices = remove_colinear(sorted_points.tolist())
    # ordered_outline_vertices = remove_colinear(ordered_vertices.tolist())
    ordered_outline_vertices = remove_colinear(polyline_vertices.tolist())
    ordered_outline_vertices = np.where(
        np.abs(ordered_outline_vertices) < 1e-4, 0, ordered_outline_vertices
    )

    return ordered_outline_vertices, normal_index


def define_cutting_points(
    bb_min: np.ndarray,
    bb_max: np.ndarray,
    poly_edges: List[Tuple[int, int]],
    regions_intervals: ZoningModel,
    ordered_outline_vertices: np.ndarray,
) -> PointsInterceptor:
    """Define cutting points the intersection of poly edges and regions intervals (x, y and z)

    Args:
        bb_min (np.ndarray): Mesh bounding box minimum
        bb_max (np.ndarray): Mesh bounding box maximum
        poly_edges (List[Tuple[int, int]]): List of polyline edges that connect point indices
        regions_intervals (ZoningModel): Region Intervals object containing x, y and z intervals
        ordered_outline_vertices (np.ndarray): Ordered outline vertices

    Returns:
        PointsInterceptor: Data structure to hold the cutting points information
    """
    points_interceptor = PointsInterceptor()
    for edge in poly_edges:
        cutting_x_int = [
            xi
            for xi in regions_intervals.x_intervals
            if is_in_range(
                xi, ordered_outline_vertices[edge[0]][0], ordered_outline_vertices[edge[1]][0]
            )
            and bb_max[0] > xi > bb_min[0]
        ]
        cutting_y_int = [
            yi
            for yi in regions_intervals.y_intervals
            if is_in_range(
                yi, ordered_outline_vertices[edge[0]][1], ordered_outline_vertices[edge[1]][1]
            )
            and bb_max[1] > yi > bb_min[1]
        ]
        cutting_z_int = [
            zi
            for zi in regions_intervals.z_intervals
            if is_in_range(
                zi, ordered_outline_vertices[edge[0]][2], ordered_outline_vertices[edge[1]][2]
            )
            and bb_max[2] > zi > bb_min[2]
        ]
        for x_int in cutting_x_int:
            cutting_point = line_plane_intersection(
                P0=tuple([x_int, 0, 0]),
                N=tuple([1, 0, 0]),
                L1=ordered_outline_vertices[edge[0]],
                L2=ordered_outline_vertices[edge[1]],
            )
            if not x_int in points_interceptor.x.keys():
                points_interceptor.x[x_int] = {}
            points_interceptor.x[x_int][edge] = cutting_point
        for y_int in cutting_y_int:
            cutting_point = line_plane_intersection(
                P0=tuple([0, y_int, 0]),
                N=tuple([0, 1, 0]),
                L1=ordered_outline_vertices[edge[0]],
                L2=ordered_outline_vertices[edge[1]],
            )
            if not y_int in points_interceptor.y.keys():
                points_interceptor.y[y_int] = {}
            points_interceptor.y[y_int][edge] = cutting_point
        for z_int in cutting_z_int:
            cutting_point = line_plane_intersection(
                P0=tuple([0, 0, z_int]),
                N=tuple([0, 0, 1]),
                L1=ordered_outline_vertices[edge[0]],
                L2=ordered_outline_vertices[edge[1]],
            )
            if not z_int in points_interceptor.z.keys():
                points_interceptor.z[z_int] = {}
            points_interceptor.z[z_int][edge] = cutting_point

    return points_interceptor


def project_grid_intersection_points(points_interceptor: PointsInterceptor) -> np.ndarray:
    """Define a grid intersection point and project to surface plane

    Args:
        points_interceptor (PointsInterceptor): Data structure to hold the cutting points information

    Returns:
        np.ndarray: Array of grid intersection points projected to surface plane
    """
    grid_intersection_points = []
    for axis_combination in [("x", "y"), ("y", "z"), ("x", "z")]:
        left_axis = points_interceptor.get_axis_dict(axis=axis_combination[0])
        right_axis = points_interceptor.get_axis_dict(axis=axis_combination[1])
        intersecting_keys = list(product(left_axis.keys(), right_axis.keys()))
        for intersection_coordinates in intersecting_keys:
            line1 = tuple(left_axis[intersection_coordinates[0]].values())
            line2 = tuple(right_axis[intersection_coordinates[1]].values())
            intersection_p = find_intersection_point(
                remove_duplicates_3d_points(line1), remove_duplicates_3d_points(line2)
            )
            grid_intersection_points.append(intersection_p)

    return np.array(grid_intersection_points)


def combine_and_sort_full_vertices(
    intersecting_vertices: np.ndarray,
    ordered_outline_vertices: np.ndarray,
    grid_intersection_points: np.ndarray,
    normal_index: int,
) -> np.ndarray:
    """Combine cutting vertices with projected grid intersection points and sort accordingly to x, y and z coordinates

    Args:
        intersecting_vertices (np.ndarray): Array of cutting vertices
        ordered_outline_vertices (np.ndarray): Array of ordered clockwise outline vertices
        grid_intersection_points (np.ndarray): Array of grid intersection points
        normal_index (int): Normal index. (1 refers to x, 2 to y and 3 refers to z, while 0 refers to inclined axis)

    Returns:
        np.ndarray: Sorted array of combined vertices
    """
    # Manually correct mesh deformation for Vendramini Case, galpao_back and galpao_front
    if normal_index == 1:
        projected_y_points = ordered_outline_vertices[:, 1]
        for point in ordered_outline_vertices:
            # Count the number of times the vertex's y-value appears among the y-values.
            counts = np.count_nonzero(np.isclose(projected_y_points, point[1], rtol=1e-3))
            if counts <= 1:
                # Find the rows that are not equal to the point_to_remove
                mask = ~np.all(ordered_outline_vertices == point, axis=1)
                ordered_outline_vertices = ordered_outline_vertices[mask]

    # outline_full_vertices = np.concatenate([intersecting_vertices, ordered_outline_vertices])
    outline_full_vertices = (
        ordered_outline_vertices
        if len(intersecting_vertices) == 0
        else np.concatenate([intersecting_vertices, ordered_outline_vertices])
    )
    full_vertices = (
        outline_full_vertices
        if len(grid_intersection_points) == 0
        else np.concatenate([outline_full_vertices, grid_intersection_points])
    )
    # Get separate 1D arrays for each coordinate
    z = full_vertices[:, 2]
    y = full_vertices[:, 1]
    x = full_vertices[:, 0]
    # Use lexsort to get the indices that would sort the vertices array
    if normal_index == 1:
        indices = np.lexsort((x, z, y))
    else:
        indices = np.lexsort((z, y, x))
    # Sort vertices with the obtained indices
    sorted_vertices = full_vertices[indices]

    return sorted_vertices


def triangulate_point_cloud(sorted_vertices: np.ndarray, normal_index: int) -> np.ndarray:
    """Creates triangulation for point cloud

    Args:
        sorted_vertices (np.ndarray): Array of sorted vertices in x, y and z axis
        normal_index (int): Normal index. (1 refers to x, 2 to y and 3 refers to z, while 0 refers to inclined axis)

    Returns:
        np.ndarray: Array of triangulated vertices, with point indices
    """
    if normal_index == 1:
        tri = Delaunay(sorted_vertices[:, 1:])  # CHANGE AXIS X NORMAL
    elif normal_index == 2:
        tri = Delaunay(sorted_vertices[:, ::2])
    else:
        tri = Delaunay(sorted_vertices[:, :2])

    return tri.simplices
