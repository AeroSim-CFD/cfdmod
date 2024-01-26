import numpy as np
from lnas import LnasFormat, LnasGeometry

from cfdmod.use_cases.pressure.force.Cf_config import CfConfig
from cfdmod.use_cases.pressure.geometry import GeometryData
from cfdmod.use_cases.pressure.zoning.body_config import BodyConfig


def get_geometry_data(body_cfg: BodyConfig, cfg: CfConfig, mesh: LnasFormat) -> GeometryData:
    """Builds a GeometryData from the mesh and the configurations

    Args:
        body_cfg (BodyConfig): Body configuration with surface list
        cfg (CfConfig): Force coefficient configuration
        mesh (LnasFormat): Input mesh

    Returns:
        GeometryData: Filtered GeometryData
    """
    geom, geometry_idx = mesh.geometry_from_list_surfaces(surfaces_names=body_cfg.surfaces)

    return GeometryData(mesh=geom, zoning_to_use=cfg.sub_bodies, triangles_idxs=geometry_idx)


def get_representative_areas(
    input_mesh: LnasGeometry, point_idx: np.ndarray
) -> tuple[float, float, float]:
    """Calculates the representative areas from the bounding box of a given mesh

    Args:
        input_mesh (LnasGeometry): Input LNAS mesh
        point_idx (np.ndarray): Array of triangle indices of each sub region

    Returns:
        tuple[float, float, float]: Representative areas tuple (Ax, Ay, Az)
    """
    geom_verts = input_mesh.triangle_vertices[point_idx].reshape(-1, 3)
    x_min, x_max = geom_verts[:, 0].min(), geom_verts[:, 0].max()
    y_min, y_max = geom_verts[:, 1].min(), geom_verts[:, 1].max()
    z_min, z_max = geom_verts[:, 2].min(), geom_verts[:, 2].max()

    Lx = x_max - x_min
    Ly = y_max - y_min
    Lz = z_max - z_min

    # Threshold to avoid big coefficients
    Lx = 1 if Lx < 1 else Lx
    Ly = 1 if Ly < 1 else Ly
    Lz = 1 if Lz < 1 else Lz

    Ax = Ly * Lz
    Ay = Lx * Lz
    Az = Lx * Ly

    return Ax, Ay, Az
