import numpy as np
from nassu.lnas import LagrangianGeometry

from cfdmod.use_cases.pressure.shape.zoning_config import ZoningModel


def triangulate_tri(sorted_vertices: np.ndarray, insertion_indices: list[int]) -> np.ndarray:
    """Triangulates a point cloud of a triangle

    Args:
        sorted_vertices (np.ndarray): Triangle vertices ordered
        insertion_indices (list[int]): Indices of the vertices inserted in the slice

    Returns:
        np.ndarray: Array of triangles
    """
    tri_indexes = []
    if len(insertion_indices) == 1:
        i = insertion_indices[0]
        tri_indexes.append([i - 1, i, (i + 2) % 4])
        tri_indexes.append([i, (i + 1) % 4, (i + 2) % 4])
    elif len(insertion_indices) == 2:
        i, j = insertion_indices[0], insertion_indices[1]
        tri_indexes.append([4, 0, 1])
        if j == 3:
            tri_indexes.append([1, 2, 3])
            tri_indexes.append([3, 4, 1])
        else:
            tri_indexes.append([1, 2, 4])
            tri_indexes.append([2, 3, 4])
    else:
        tri_indexes.append([0, 1, 2])

    return sorted_vertices[np.array(tri_indexes)]


def slice_triangle(tri_verts: np.ndarray, axis: int, axis_value: float) -> np.ndarray:
    """Slice a triangle from a given plane

    Args:
        tri_verts (np.ndarray): Vertices of the triangle to slice
        axis (int): Axis index (x=0, y=1, z=2)
        axis_value (float): Value of the interval

    Returns:
        np.ndarray: Array of triangle vertices resulted from slicing
    """
    intersected_pts = tri_verts.copy()
    insertion_indices = []

    for i in range(3):
        if len(intersected_pts) > 4:
            # Sliced all possible lines
            continue
        else:
            p1, p2 = tri_verts[i], tri_verts[(i + 1) % 3]

            if (p1[axis] < axis_value and p2[axis] > axis_value) or (
                p1[axis] > axis_value and p2[axis] < axis_value
            ):
                t = (axis_value - p1[axis]) / (p2[axis] - p1[axis])
                intersect_pt = p1 + t * (p2 - p1)

                insert_idx = i + 1 + intersected_pts.shape[0] // 4
                insertion_indices.append(insert_idx)
                intersected_pts = np.insert(intersected_pts, insert_idx, intersect_pt, axis=0)

    if len(intersected_pts) == 3:
        return np.array([tri_verts])
    else:
        return triangulate_tri(intersected_pts, sorted(insertion_indices))


def slice_surface(surface: LagrangianGeometry, axis: int, interval: float) -> LagrangianGeometry:
    """From a given plane, slice the surface's triangles

    Args:
        surface (LagrangianGeometry): Input LNAS surface mesh
        axis (int): Axis index (x=0, y=1, z=2)
        interval (float): Value of the interval

    Returns:
        LagrangianGeometry: Sliced LNAS surface mesh
    """
    new_triangles = np.zeros((0, 3, 3))

    for tri_verts, tri_normal in zip(surface.triangle_vertices, surface.normals):
        if np.abs(tri_normal).max() == np.abs(tri_normal)[axis]:
            new_triangles = np.concatenate((new_triangles, [tri_verts]), axis=0)
            continue
        sliced_triangles = slice_triangle(tri_verts, axis, interval)
        new_triangles = np.concatenate((new_triangles, sliced_triangles), axis=0)

    full_verts = new_triangles.reshape(len(new_triangles) * 3, 3)
    verts, triangles = np.unique(full_verts, axis=0, return_inverse=True)

    return LagrangianGeometry(verts, triangles.reshape(-1, 3))


def get_mesh_bounds(input_mesh: LagrangianGeometry) -> tuple[tuple[float, float], ...]:
    """Calculates the bounding box of a given mesh

    Args:
        input_mesh (LagrangianGeometry): Input LNAS mesh

    Returns:
        tuple[tuple[float, float], ...]: Bounding box tuples ((x_min, x_max), (y_min, y_max), (z_min, z_max))
    """
    x_min, x_max = input_mesh.vertices[:, 0].min(), input_mesh.vertices[:, 0].max()
    y_min, y_max = input_mesh.vertices[:, 1].min(), input_mesh.vertices[:, 1].max()
    z_min, z_max = input_mesh.vertices[:, 2].min(), input_mesh.vertices[:, 2].max()

    return ((x_min, x_max), (y_min, y_max), (z_min, z_max))


def create_regions_mesh(
    input_mesh: LagrangianGeometry, regions_intervals: ZoningModel
) -> LagrangianGeometry:
    """Generates a new LagrangianGeometry mesh from intersecting regions intervals

    Args:
        input_mesh (LagrangianGeometry): Input LNAS mesh
        regions_intervals (ZoningModel): Region intervals description

    Returns:
        LagrangianGeometry: New intersected mesh
    """
    mesh_bounds = get_mesh_bounds(input_mesh)
    slicing_mesh = input_mesh.copy()

    for x_int in regions_intervals.x_intervals:
        if x_int <= mesh_bounds[0][0] or x_int >= mesh_bounds[0][1]:
            continue
        slicing_mesh = slice_surface(slicing_mesh, 0, x_int)

    for y_int in regions_intervals.y_intervals:
        if y_int <= mesh_bounds[1][0] or y_int >= mesh_bounds[1][1]:
            continue
        slicing_mesh = slice_surface(slicing_mesh, 1, y_int)

    for z_int in regions_intervals.z_intervals:
        if z_int <= mesh_bounds[2][0] or z_int >= mesh_bounds[2][1]:
            continue
        slicing_mesh = slice_surface(slicing_mesh, 2, z_int)

    return slicing_mesh
