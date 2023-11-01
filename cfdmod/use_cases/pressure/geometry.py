import numpy as np
from nassu.lnas import LagrangianFormat, LagrangianGeometry

from cfdmod.use_cases.pressure.zoning.body_config import BodyConfig


def get_excluded_surfaces(mesh: LagrangianFormat, sfc_list: list[str]) -> LagrangianGeometry:
    """Filters the surfaces that were excluded in processing

    Args:
        mesh (LagrangianFormat): LNAS body mesh
        sfc_list (list[str]): List of excluded surfaces

    Returns:
        LagrangianGeometry: Returns a LagrangianGeometry if any surface was excluded
    """
    excluded_ids = np.array([], dtype=np.uint32)
    for excluded_sfc in sfc_list:
        if not excluded_sfc in mesh.surfaces.keys():
            continue
        ids = mesh.surfaces[excluded_sfc].copy()
        excluded_ids = np.concatenate((excluded_ids, ids))

    if excluded_ids.size != 0:
        excluded_geom = LagrangianGeometry(
            vertices=mesh.geometry.vertices.copy(),
            triangles=mesh.geometry.triangles[excluded_ids].copy(),
        )
        return excluded_geom
    else:
        raise Exception("No geometry could be filtered from the list of surfaces.")


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
                raise Exception(
                    f"Surface {sfc} defined in body is not separated in the LNAS file."
                )
            geometry_idx = np.concatenate((geometry_idx, mesh.surfaces[sfc]))

    body_geom = LagrangianGeometry(
        vertices=mesh.geometry.vertices.copy(),
        triangles=mesh.geometry.triangles[geometry_idx].copy(),
    )

    return body_geom, geometry_idx
