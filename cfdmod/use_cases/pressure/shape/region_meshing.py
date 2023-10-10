import pathlib

import numpy as np
from nassu.lnas import LagrangianGeometry
from scipy.spatial import Delaunay

from cfdmod.use_cases.pressure.shape.regions import ZoningModel


def triangulate_point_cloud(sorted_vertices: np.ndarray) -> np.ndarray:
    """Creates triangulation for point cloud

    Args:
        sorted_vertices (np.ndarray): Array of sorted vertices

    Returns:
        np.ndarray: Array of triangulated vertices, with point indices
    """
    normal = np.cross(
        sorted_vertices[1] - sorted_vertices[0], sorted_vertices[2] - sorted_vertices[0]
    )
    if np.linalg.norm(normal) == 0:
        normal = np.cross(
            sorted_vertices[2] - sorted_vertices[0], sorted_vertices[3] - sorted_vertices[0]
        )
    n = normal / np.linalg.norm(normal)

    jitter = 1e-9  # Small random value
    jittering_array = np.random.uniform(-jitter, jitter, sorted_vertices.shape)
    sorted_vertices += jittering_array

    if np.allclose(n, np.array([1, 0, 0])) or np.allclose(n, np.array([-1, 0, 0])):
        tri = Delaunay(sorted_vertices[:, 1:])
    elif np.allclose(n, np.array([0, 1, 0])) or np.allclose(n, np.array([0, -1, 0])):
        tri = Delaunay(sorted_vertices[:, ::2])
    else:
        tri = Delaunay(sorted_vertices[:, :2])

    # sorted_vertices -= jittering_array

    return sorted_vertices[tri.simplices]


def slice_triangle(tri_verts: np.ndarray, axis: int, axis_value: float):
    intersected_pts = tri_verts.copy()
    for i in range(3):
        if len(intersected_pts) > 4:
            # Sliced all possible lines
            continue

        p1, p2 = tri_verts[i], tri_verts[(i + 1) % 3]

        if (p1[axis] < axis_value and p2[axis] > axis_value) or (
            p1[axis] > axis_value and p2[axis] < axis_value
        ):
            t = (axis_value - p1[axis]) / (p2[axis] - p1[axis])
            intersect_pt = p1 + t * (p2 - p1)

            insert_idx = i + 1 + intersected_pts.shape[0] // 4
            intersected_pts = np.insert(intersected_pts, insert_idx, intersect_pt, axis=0)

    if len(intersected_pts) == 3:
        return np.array([tri_verts])
    else:
        return triangulate_point_cloud(intersected_pts)


def slice_surface(surface: LagrangianGeometry, axis: int, interval: float) -> LagrangianGeometry:
    new_triangles = np.zeros((0, 3, 3))
    axis_normal = np.array([0, 0, 0])
    axis_normal[axis] = 1

    for tri_verts, tri_normal in zip(surface.triangle_vertices, surface.normals):
        if np.allclose(tri_normal, axis_normal) or np.allclose(tri_normal, -axis_normal):
            new_triangles = np.concatenate((new_triangles, [tri_verts]), axis=0)
            continue
        sliced_triangles = slice_triangle(tri_verts, axis, interval)
        new_triangles = np.concatenate((new_triangles, sliced_triangles), axis=0)

    full_verts = new_triangles.reshape(len(new_triangles) * 3, 3)
    verts, triangles = np.unique(full_verts, axis=0, return_inverse=True)

    return LagrangianGeometry(verts, triangles.reshape(-1, 3))


def get_mesh_bounds(input_mesh: LagrangianGeometry) -> tuple[tuple[float, float], ...]:
    x_min, x_max = input_mesh.vertices[:, 0].min(), input_mesh.vertices[:, 0].max()
    y_min, y_max = input_mesh.vertices[:, 1].min(), input_mesh.vertices[:, 1].max()
    z_min, z_max = input_mesh.vertices[:, 2].min(), input_mesh.vertices[:, 2].max()

    return ((x_min, x_max), (y_min, y_max), (z_min, z_max))


def create_regions_mesh(
    input_mesh: LagrangianGeometry, regions_intervals: ZoningModel
) -> LagrangianGeometry:
    mesh_bounds = get_mesh_bounds(input_mesh)
    slicing_mesh = input_mesh.copy()

    print(slicing_mesh.triangle_vertices.shape)
    for x_int in regions_intervals.x_intervals:
        if x_int <= mesh_bounds[0][0] or x_int >= mesh_bounds[0][1]:
            continue
        slicing_mesh = slice_surface(slicing_mesh, 0, x_int)

    print(slicing_mesh.triangle_vertices.shape)
    for y_int in regions_intervals.y_intervals:
        if y_int <= mesh_bounds[1][0] or y_int >= mesh_bounds[1][1]:
            continue
        slicing_mesh = slice_surface(slicing_mesh, 1, y_int)
        slicing_mesh.export_stl(pathlib.Path(f"./output/pressure/sfc.y.{y_int}.stl"))

    print(slicing_mesh.triangle_vertices.shape)
    for z_int in regions_intervals.z_intervals:
        if z_int <= mesh_bounds[2][0] or z_int >= mesh_bounds[2][1]:
            continue
        slicing_mesh = slice_surface(slicing_mesh, 2, z_int)
        slicing_mesh.export_stl(pathlib.Path(f"./output/pressure/sfc.z.{z_int}.stl"))

    print(slicing_mesh.triangle_vertices.shape)

    return slicing_mesh
