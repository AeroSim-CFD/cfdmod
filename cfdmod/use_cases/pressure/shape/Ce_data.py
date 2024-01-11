from dataclasses import dataclass
from typing import Optional

import numpy as np
import pandas as pd
from lnas import LnasFormat, LnasGeometry
from vtk import vtkPolyData

from cfdmod.api.geometry.region_meshing import create_regions_mesh
from cfdmod.api.vtk.write_vtk import create_polydata_for_cell_data
from cfdmod.logger import logger
from cfdmod.use_cases.pressure.extreme_values import ExtremeValuesParameters
from cfdmod.use_cases.pressure.path_manager import CePathManager
from cfdmod.use_cases.pressure.shape.Ce_config import CeConfig
from cfdmod.use_cases.pressure.shape.zoning_config import ZoningModel
from cfdmod.use_cases.pressure.zoning.processing import (
    calculate_statistics,
    combine_stats_data_with_mesh,
    get_indexing_mask,
)
from cfdmod.utils import create_folders_for_file


@dataclass
class RawSurfaceData:
    sfc_mesh: LnasGeometry
    zoning_to_use: ZoningModel
    sfc_triangles_idxs: np.ndarray


@dataclass
class ProcessedSurfaceData:
    df_regions: pd.DataFrame
    surface_ce: pd.DataFrame
    surface_ce_stats: pd.DataFrame
    regions_mesh: LnasGeometry
    region_data_df: pd.DataFrame
    polydata: vtkPolyData

    def save_outputs(self, sfc_label: str, cfg_label: str, path_manager: CePathManager):
        # Output 1: Ce_regions
        regions_path = path_manager.get_regions_df_path(sfc_label, cfg_label)
        create_folders_for_file(regions_path)
        self.df_regions.to_hdf(regions_path, key="Regions", mode="w", index=False)

        # Output 2: Ce(t)
        timeseries_path = path_manager.get_timeseries_df_path(sfc_label, cfg_label)
        create_folders_for_file(timeseries_path)
        self.surface_ce.to_hdf(timeseries_path, key="Ce_t", mode="w", index=False)

        # Output 3: Ce_stats
        stats_path = path_manager.get_stats_df_path(sfc_label, cfg_label)
        create_folders_for_file(stats_path)
        self.surface_ce_stats.to_hdf(stats_path, key="Ce_stats", mode="w", index=False)

        # Output 4: Regions Mesh
        mesh_path = path_manager.get_surface_path(sfc_label=sfc_label, cfg_label=cfg_label)
        create_folders_for_file(mesh_path)
        self.regions_mesh.export_stl(mesh_path)


def get_surfaces_raw_data(
    surface_dict: dict[str, list[str]], cfg: CeConfig, mesh: LnasFormat
) -> dict[str, RawSurfaceData]:
    """Get surfaces raw data from mesh

    Args:
        surface_dict (dict[str, list[str]]): Dictionary with surface list keyed by surface label
        cfg (CeConfig): Post processing configuration
        mesh (LnasFormat): LNAS mesh

    Returns:
        dict[str, RawSurfaceData]: Dictionary with raw surface data keyed by surface label
    """
    raw_surfaces: dict[str, RawSurfaceData] = {}
    for sfc_lbl, sfc_list in surface_dict.items():
        if sfc_lbl in cfg.zoning.exclude:  # type: ignore (already validated in class)
            logger.info(f"Surface {sfc_lbl} ignored!")  # Ignore surface
            continue
        surface_geom, sfc_triangles_idxs = filter_surface_geometry(mesh=mesh, sfc_list=sfc_list)
        raw_surface = build_surface_raw_data(
            sfc_mesh=surface_geom,
            sfc_label=sfc_lbl,
            cfg=cfg,
            sfc_triangles_idxs=sfc_triangles_idxs,
        )
        raw_surfaces[sfc_lbl] = raw_surface
    return raw_surfaces


def filter_surface_geometry(
    mesh: LnasFormat, sfc_list: list[str]
) -> tuple[LnasGeometry, np.ndarray]:
    """Filter body from surface list

    Args:
        mesh (LnasFormat): LNAS mesh with every surface available
        sfc_list (list[str]): List of surfaces to be filtered

    Returns:
        tuple[LnasGeometry, np.ndarray]: Tuple with filtered LNAS mesh geometry
        and the filtered triangle indices
    """
    sfc_mesh = LnasGeometry(
        vertices=mesh.geometry.vertices, triangles=np.empty((0, 3), dtype=np.uint32)
    )
    sfc_triangles_idxs = np.array([])

    for sfc in sfc_list:
        m = mesh.geometry_from_surface(sfc)
        sfc_mesh.triangles = np.vstack((sfc_mesh.triangles, m.triangles))
        sfc_triangles_idxs = np.hstack((sfc_triangles_idxs, mesh.surfaces[sfc].copy()))

    return sfc_mesh, sfc_triangles_idxs


def build_surface_raw_data(
    sfc_mesh: LnasGeometry, sfc_label: str, cfg: CeConfig, sfc_triangles_idxs: np.ndarray
) -> RawSurfaceData:
    """Builds a raw surface data object

    Args:
        sfc_mesh (LnasGeometry): Geometry of the surface
        sfc_label (str): Surface label
        cfg (CeConfig): Post processing configuration
        sfc_triangles_idxs (np.ndarray): Filtered triangle indices

    Returns:
        RawSurfaceData: Raw surface data object
    """
    zoning_to_use = get_surface_zoning(sfc_mesh, sfc_label, cfg)

    return RawSurfaceData(
        sfc_mesh=sfc_mesh, zoning_to_use=zoning_to_use, sfc_triangles_idxs=sfc_triangles_idxs
    )


def process_surface(
    raw_surface: RawSurfaceData,
    cfg: CeConfig,
    cp_data: pd.DataFrame,
    extreme_params: Optional[ExtremeValuesParameters] = None,
) -> ProcessedSurfaceData:
    """Filters a surface from the body and processes it

    Args:
        raw_surface (RawSurfaceData): Raw surface to process
        cfg (CeConfig): Post processing configuration
        cp_data (pd.DataFrame): Pressure coefficients DataFrame
        extreme_params (Optional[ExtremeValuesParameters]): Parameters for extreme values analysis. Defaults to None.

    Returns:
        ProcessedSurfaceData: Processed Surface Data object
    """
    df_regions = raw_surface.zoning_to_use.get_regions_df()

    transformed_surface = raw_surface.sfc_mesh.copy()
    transformed_surface.apply_transformation(cfg.transformation.get_geometry_transformation())

    triangles_region = get_indexing_mask(mesh=transformed_surface, df_regions=df_regions)
    surface_ce = transform_to_Ce(
        surface_mesh=raw_surface.sfc_mesh,
        cp_data=cp_data,
        triangles_region=triangles_region,
    )

    surface_ce_stats = calculate_statistics(
        surface_ce,
        statistics_to_apply=cfg.statistics,
        variables=["Ce"],
        group_by_key="region_idx",
        extreme_params=extreme_params,
    )

    regions_mesh = create_regions_mesh(
        transformed_surface,
        (
            raw_surface.zoning_to_use.x_intervals,
            raw_surface.zoning_to_use.y_intervals,
            raw_surface.zoning_to_use.z_intervals,
        ),
    )

    regions_mesh_triangles_region = get_indexing_mask(mesh=regions_mesh, df_regions=df_regions)
    regions_mesh.apply_transformation(
        cfg.transformation.get_geometry_transformation(), invert_transf=True
    )

    region_data_df = combine_stats_data_with_mesh(
        regions_mesh, regions_mesh_triangles_region, surface_ce_stats
    )
    if (region_data_df.isnull().sum() != 0).any():
        logger.warning("Region refinement is greater than data refinement. Resulted in NaN values")

    polydata = create_polydata_for_cell_data(region_data_df, regions_mesh)

    return ProcessedSurfaceData(
        df_regions=df_regions,
        surface_ce=surface_ce,
        surface_ce_stats=surface_ce_stats,
        regions_mesh=regions_mesh,
        region_data_df=region_data_df,
        polydata=polydata,
    )


def get_surface_zoning(mesh: LnasGeometry, sfc: str, config: CeConfig) -> ZoningModel:
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


def transform_to_Ce(
    surface_mesh: LnasGeometry,
    cp_data: pd.DataFrame,
    triangles_region: np.ndarray,
) -> pd.DataFrame:
    """Transforms pressure coefficient for surface to shape coefficient

    Args:
        surface_mesh (LnasGeometry): Surface mesh
        cp_data (pd.DataFrame): Body pressure coefficient data
        triangles_region (np.ndarray): Surface triangles region indexing

    Returns:
        pd.DataFrame: Shape coefficient for surface
    """
    n_timesteps = cp_data["time_step"].nunique()
    triangles_areas = surface_mesh.areas.copy()

    surface_cp = cp_data.copy()

    surface_cp["region_idx"] = np.tile(triangles_region, n_timesteps)
    surface_cp["tri_area"] = np.tile(triangles_areas, n_timesteps)
    surface_cp["f/q"] = surface_cp["cp"] * surface_cp["tri_area"]

    surface_ce = (
        surface_cp.groupby(["region_idx", "time_step"])  # type: ignore
        .agg(
            total_area=pd.NamedAgg(column="tri_area", aggfunc="sum"),
            total_force=pd.NamedAgg(column="f/q", aggfunc="sum"),
        )
        .reset_index()
    )

    surface_ce["Ce"] = surface_ce["total_force"] / surface_ce["total_area"]

    return surface_ce
