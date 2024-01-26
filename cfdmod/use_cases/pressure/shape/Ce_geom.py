import numpy as np
from lnas import LnasFormat, LnasGeometry

from cfdmod.api.geometry.region_meshing import create_regions_mesh
from cfdmod.logger import logger
from cfdmod.use_cases.pressure.geometry import GeometryData
from cfdmod.use_cases.pressure.shape.Ce_config import CeConfig
from cfdmod.use_cases.pressure.shape.zoning_config import ZoningModel
from cfdmod.use_cases.pressure.zoning.processing import get_indexing_mask


def _get_surface_zoning(mesh: LnasGeometry, sfc: str, config: CeConfig) -> ZoningModel:
    """Get the surface respective zoning configuration

    Args:
        mesh (LnasGeometry): Surface LNAS mesh
        sfc (str): Surface label
        config (CeConfig): Post process configuration

    Returns:
        ZoningModel: Zoning configuration
    """
    if sfc in config.zoning.no_zoning:  # type: ignore
        zoning = ZoningModel(**{})
    elif sfc in config.zoning.surfaces_in_exception:  # type: ignore
        zoning = [cfg for cfg in config.zoning.exceptions.values() if sfc in cfg.surfaces][0]  # type: ignore
    else:
        zoning = config.zoning.global_zoning  # type: ignore
        if len(np.unique(np.round(mesh.normals, decimals=2), axis=0)) == 1:
            ignore_axis = np.where(np.abs(mesh.normals[0]) == np.abs(mesh.normals[0]).max())[0][0]
            zoning = zoning.ignore_axis(ignore_axis)

    return zoning.offset_limits(0.1)


def get_geometry_data(
    surface_dict: dict[str, list[str]], cfg: CeConfig, mesh: LnasFormat
) -> dict[str, GeometryData]:
    """Get surfaces geometry data from mesh

    Args:
        surface_dict (dict[str, list[str]]): Dictionary with surface list keyed by surface label
        cfg (CeConfig): Post processing configuration
        mesh (LnasFormat): LNAS mesh

    Returns:
        dict[str, GeometryData]: Dictionary with geometry data keyed by surface label
    """
    geom_dict: dict[str, GeometryData] = {}
    for sfc_lbl, sfc_list in surface_dict.items():
        if sfc_lbl in cfg.zoning.exclude:  # type: ignore (already validated in class)
            logger.info(f"Surface {sfc_lbl} ignored!")  # Ignore surface
            continue
        surface_geom, sfc_triangles_idxs = mesh.geometry_from_list_surfaces(
            surfaces_names=sfc_list
        )
        zoning_to_use = _get_surface_zoning(mesh=surface_geom, sfc=sfc_lbl, config=cfg)

        geom_data = GeometryData(
            mesh=surface_geom,
            zoning_to_use=zoning_to_use,
            triangles_idxs=sfc_triangles_idxs,
        )
        geom_dict[sfc_lbl] = geom_data
    return geom_dict


def generate_regions_mesh(
    geom_data: GeometryData, cfg: CeConfig
) -> tuple[LnasGeometry, np.ndarray]:
    """Generates a new mesh intersecting the input mesh with the regions definition

    Args:
        geom_data (GeometryData): Geometry data with surface mesh and regions information
        cfg (CeConfig): Shape coefficient configuration

    Returns:
        tuple[LnasGeometry, np.ndarray]: Tuple with region mesh and region mesh triangle indexing
    """
    transformed_surface = geom_data.mesh.copy()
    transformed_surface.apply_transformation(cfg.transformation.get_geometry_transformation())

    regions_mesh = create_regions_mesh(
        transformed_surface,
        (
            geom_data.zoning_to_use.x_intervals,
            geom_data.zoning_to_use.y_intervals,
            geom_data.zoning_to_use.z_intervals,
        ),
    )

    df_regions = geom_data.zoning_to_use.get_regions_df()
    regions_mesh_triangles_indexing = get_indexing_mask(mesh=regions_mesh, df_regions=df_regions)

    regions_mesh.apply_transformation(
        cfg.transformation.get_geometry_transformation(), invert_transf=True
    )

    return regions_mesh, regions_mesh_triangles_indexing
