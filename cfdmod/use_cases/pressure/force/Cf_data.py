import numpy as np
from nassu.lnas import LagrangianFormat, LagrangianGeometry

from cfdmod.use_cases.pressure.force.body_config import BodyConfig


def get_geometry_from_mesh(
    body_cfg: BodyConfig, mesh: LagrangianFormat
) -> tuple[LagrangianGeometry, np.ndarray]:
    """Filters the mesh from the list of surfaces that define the body in config

    Args:
        body_cfg (BodyConfig): Body configuration
        mesh (LagrangianFormat): LNAS mesh

    Raises:
        Exception: Surface specified is not defined in LNAS

    Returns:
        tuple[LagrangianGeometry, np.ndarray]: Tuple containing the body geometry and the filtered triangle indexes
    """
    if len(body_cfg.surfaces) == 0:
        # Include all surfaces
        geometry_idx = np.arange(0, len(mesh.geometry.triangles))
    else:
        # Filter mesh for all surfaces
        geometry_idx = np.array([], dtype=np.int32)
        for sfc in body_cfg.surfaces:
            if sfc not in mesh.surfaces.keys():
                raise Exception("Surface defined in body is not separated in the LNAS file.")
            geometry_idx = np.concatenate((geometry_idx, mesh.surfaces[sfc]))

    body_geom = LagrangianGeometry(
        vertices=mesh.geometry.vertices.copy(),
        triangles=mesh.geometry.triangles[geometry_idx].copy(),
    )

    return body_geom, geometry_idx


def get_representative_areas(input_mesh: LagrangianGeometry) -> tuple[float, float, float]:
    """Calculates the representative areas from the bounding box of a given mesh

    Args:
        input_mesh (LagrangianGeometry): Input LNAS mesh

    Returns:
        tuple[float, float, float]: Representative areas tuple (Ax, Ay, Az)
    """
    x_min, x_max = input_mesh.vertices[:, 0].min(), input_mesh.vertices[:, 0].max()
    y_min, y_max = input_mesh.vertices[:, 1].min(), input_mesh.vertices[:, 1].max()
    z_min, z_max = input_mesh.vertices[:, 2].min(), input_mesh.vertices[:, 2].max()

    Lx = x_max - x_min
    Ly = y_max - y_min
    Lz = z_max - z_min

    Ax = Ly * Lz
    Ay = Lx * Lz
    Az = Lx * Ly

    return Ax, Ay, Az
