from dataclasses import dataclass

import numpy as np
import pandas as pd
from nassu.lnas import LagrangianFormat, LagrangianGeometry
from vtk import vtkPolyData

from cfdmod.api.geometry.region_meshing import create_regions_mesh
from cfdmod.api.vtk.write_vtk import create_polydata_for_cell_data
from cfdmod.logger import logger
from cfdmod.use_cases.pressure.path_manager import CePathManager
from cfdmod.use_cases.pressure.shape.Ce_config import CeConfig
from cfdmod.use_cases.pressure.shape.zoning_config import ZoningModel
from cfdmod.use_cases.pressure.zoning.processing import (
    calculate_statistics,
    combine_stats_data_with_mesh,
    get_indexing_mask,
)


@dataclass
class ProcessedSurfaceData:
    df_regions: pd.DataFrame
    surface_ce: pd.DataFrame
    surface_ce_stats: pd.DataFrame
    regions_mesh: LagrangianGeometry
    region_data_df: pd.DataFrame
    polydata: vtkPolyData

    def save_outputs(self, sfc_label: str, cfg_label: str, path_manager: CePathManager):
        # Output 1: Ce_regions
        self.df_regions.to_hdf(
            path_manager.get_regions_df_path(sfc_label, cfg_label),
            key="Regions",
            mode="w",
            index=False,
        )

        # Output 2: Ce(t)
        self.surface_ce.to_hdf(
            path_manager.get_timeseries_df_path(sfc_label, cfg_label),
            key="Ce_t",
            mode="w",
            index=False,
        )

        # Output 3: Ce_stats
        self.surface_ce_stats.to_hdf(
            path_manager.get_stats_df_path(sfc_label, cfg_label),
            key="Ce_stats",
            mode="w",
            index=False,
        )

        # Output 4: Regions Mesh
        self.regions_mesh.export_stl(
            path_manager.get_surface_path(sfc_label=sfc_label, cfg_label=cfg_label)
        )


def process_surface(
    body_mesh: LagrangianFormat,
    sfc_label: str,
    cfg: CeConfig,
    cp_data: pd.DataFrame,
    n_timesteps: int,
) -> ProcessedSurfaceData:
    """Filters a surface from the body and processes it

    Args:
        body_mesh (LagrangianFormat): LNAS body containing its surfaces
        sfc_label (str): Surface label to be processed
        cfg (CeConfig): Post processing configuration
        cp_data (pd.DataFrame): Pressure coefficients DataFrame
        n_timesteps (int): Number of timesteps to be processed

    Returns:
        ProcessedSurfaceData: Processed Surface Data object
    """
    sfc_mesh = body_mesh.geometry_from_surface(sfc_label)
    zoning_to_use = get_surface_zoning(sfc_mesh, sfc_label, cfg)

    df_regions = zoning_to_use.get_regions_df()

    triangles_region = get_indexing_mask(mesh=sfc_mesh, df_regions=df_regions)
    sfc_triangles_idxs = body_mesh.surfaces[sfc_label].copy()
    surface_ce = transform_to_Ce(
        surface_mesh=sfc_mesh,
        cp_data=cp_data,
        sfc_triangles_idxs=sfc_triangles_idxs,
        triangles_region=triangles_region,
        n_timesteps=n_timesteps,
    )

    surface_ce_stats = calculate_statistics(
        surface_ce, statistics_to_apply=cfg.statistics, variables=["Ce"]
    )

    regions_mesh = create_regions_mesh(
        sfc_mesh, (zoning_to_use.x_intervals, zoning_to_use.y_intervals, zoning_to_use.z_intervals)
    )
    regions_mesh_triangles_region = get_indexing_mask(mesh=regions_mesh, df_regions=df_regions)

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


def get_surface_zoning(mesh: LagrangianGeometry, sfc: str, config: CeConfig) -> ZoningModel:
    """Get the surface respective zoning configuration

    Args:
        mesh (LagrangianGeometry): Surface LNAS mesh
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
    surface_mesh: LagrangianGeometry,
    cp_data: pd.DataFrame,
    sfc_triangles_idxs: np.ndarray,
    triangles_region: np.ndarray,
    n_timesteps: int,
) -> pd.DataFrame:
    """Transforms pressure coefficient for surface to shape coefficient

    Args:
        surface_mesh (LagrangianGeometry): Surface mesh
        cp_data (pd.DataFrame): Body pressure coefficient data
        sfc_triangles_idxs (np.ndarray): Surface triangles index from body mesh
        triangles_region (np.ndarray): Surface triangles region indexing
        n_timesteps (int): Number of timesteps in data

    Returns:
        pd.DataFrame: Shape coefficient for surface
    """

    triangles_areas = surface_mesh.areas.copy()

    surface_cp = cp_data[cp_data["point_idx"].isin(sfc_triangles_idxs)].copy()

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
