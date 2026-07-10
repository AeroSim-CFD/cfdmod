import warnings

import numpy as np
from lnas import LnasGeometry

# The pure-numpy slicing primitives live in the dependency-light
# ``cfdmod.geometry.triangle_slicing`` so the v3 op layer can import them
# without dragging in ``cfdmod.io`` (h5py / pandas / ...). Re-exported here
# for backwards compatibility with existing callers and tests.
from cfdmod.geometry.triangle_slicing import slice_triangle, triangulate_tri

__all__ = [
    "triangulate_tri",
    "slice_triangle",
    "clean_triangles",
    "slice_surface",
    "get_mesh_bounds",
    "create_regions_mesh",
]


def clean_triangles(geom: LnasGeometry, minimal_area: float = 1e-5) -> LnasGeometry:
    """Removes any malformed triangles from the geometry

    Args:
        geom (LnasGeometry): Geometry to be cleaned

    Returns:
        LnasGeometry: Filtered geometry with all valid triangles
    """
    cross_prod = geom._cross_prod()
    norm_cross_prod = np.linalg.norm(cross_prod, axis=1)

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        normals = cross_prod / norm_cross_prod[:, np.newaxis]

    areas = norm_cross_prod / 2
    nan_normals = ~np.isnan(normals)
    filter_areas = areas > minimal_area

    idxs_triangles = (
        geom.triangles[np.all(nan_normals, axis=1) & filter_areas].copy().reshape(-1, 3)
    )
    cleaned_geom = LnasGeometry(vertices=geom.vertices.copy(), triangles=idxs_triangles)
    cleaned_geom._full_update()

    return cleaned_geom


def slice_surface(surface: LnasGeometry, axis: int, interval: float) -> LnasGeometry:
    """From a given plane, slice the surface's triangles

    Args:
        surface (LnasGeometry): Input LNAS surface mesh
        axis (int): Axis index (x=0, y=1, z=2)
        interval (float): Value of the interval

    Returns:
        LnasGeometry: Sliced LNAS surface mesh
    """
    triangles_list = []

    for tri_verts, tri_normal in zip(surface.triangle_vertices, surface.normals):
        # If triangle normal is the same of plane normal, not slice it
        if np.abs(tri_normal).max() == np.abs(tri_normal)[axis]:
            triangles_list.extend([tri_verts.tolist()])
            continue
        if tri_verts[:, axis].max() < interval or tri_verts[:, axis].min() > interval:
            triangles_list.extend([tri_verts.tolist()])
        else:
            sliced_triangles = slice_triangle(tri_verts, axis, interval)
            triangles_list.extend(sliced_triangles.tolist())

    new_triangles = np.array(triangles_list, dtype=np.float32)

    full_verts = new_triangles.reshape(len(triangles_list) * 3, 3)
    verts, triangles = np.unique(full_verts, axis=0, return_inverse=True)

    geom = LnasGeometry(verts, triangles.reshape(-1, 3))
    geom = clean_triangles(geom=geom)

    return geom


def get_mesh_bounds(input_mesh: LnasGeometry) -> tuple[tuple[float, float], ...]:
    """Calculates the bounding box of a given mesh

    Args:
        input_mesh (LnasGeometry): Input LNAS mesh

    Returns:
        tuple[tuple[float, float], ...]: Bounding box tuples ((x_min, x_max), (y_min, y_max), (z_min, z_max))
    """
    x_min, x_max = input_mesh.vertices[:, 0].min(), input_mesh.vertices[:, 0].max()
    y_min, y_max = input_mesh.vertices[:, 1].min(), input_mesh.vertices[:, 1].max()
    z_min, z_max = input_mesh.vertices[:, 2].min(), input_mesh.vertices[:, 2].max()

    return ((x_min, x_max), (y_min, y_max), (z_min, z_max))


def create_regions_mesh(
    input_mesh: LnasGeometry, intervals: tuple[list[float], ...]
) -> LnasGeometry:
    """Generates a new LnasGeometry mesh from intersecting intervals

    Args:
        input_mesh (LnasGeometry): Input LNAS mesh
        intervals (tuple[list[float], ...]): List of intervals in each axis

    Returns:
        LnasGeometry: New intersected mesh
    """
    mesh_bounds = get_mesh_bounds(input_mesh)
    slicing_mesh = input_mesh.copy()

    for x_int in intervals[0]:
        if x_int <= mesh_bounds[0][0] or x_int >= mesh_bounds[0][1]:
            continue
        slicing_mesh = slice_surface(slicing_mesh, 0, x_int)

    for y_int in intervals[1]:
        if y_int <= mesh_bounds[1][0] or y_int >= mesh_bounds[1][1]:
            continue
        slicing_mesh = slice_surface(slicing_mesh, 1, y_int)

    for z_int in intervals[2]:
        if z_int <= mesh_bounds[2][0] or z_int >= mesh_bounds[2][1]:
            continue
        slicing_mesh = slice_surface(slicing_mesh, 2, z_int)

    return slicing_mesh
