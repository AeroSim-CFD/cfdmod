import pathlib
from dataclasses import dataclass

import numpy as np
import pandas as pd
from lnas import LnasFormat, LnasGeometry

from cfdmod.api.vtk.write_vtk import create_polydata_for_cell_data
from cfdmod.logger import logger
from cfdmod.use_cases.pressure.chunking import process_timestep_groups
from cfdmod.use_cases.pressure.geometry import (
    GeometryData,
    ProcessedEntity,
    get_excluded_entities,
    tabulate_geometry_data,
)
from cfdmod.use_cases.pressure.output import CommonOutput
from cfdmod.use_cases.pressure.path_manager import CePathManager
from cfdmod.use_cases.pressure.shape.Ce_config import CeConfig
from cfdmod.use_cases.pressure.shape.Ce_geom import generate_regions_mesh, get_geometry_data
from cfdmod.use_cases.pressure.zoning.processing import (
    calculate_statistics,
    combine_stats_data_with_mesh,
)
from cfdmod.utils import convert_dataframe_into_matrix, create_folders_for_file


@dataclass
class CeOutput(CommonOutput):
    def export_mesh(self, cfg_label: str, path_manager: CePathManager):
        # Regions Mesh
        mesh_path = path_manager.get_surface_path(cfg_lbl=cfg_label, sfc_lbl="body")
        create_folders_for_file(mesh_path)
        regions_mesh = self.processed_entities[0].mesh.copy()
        regions_mesh.join([sfc.mesh.copy() for sfc in self.processed_entities[1:]])
        regions_mesh.export_stl(mesh_path)

        # (Optional) Excluded Mesh
        if len(self.excluded_entities) != 0:
            excluded_mesh_path = path_manager.get_surface_path(
                cfg_lbl=cfg_label, sfc_lbl="excluded_surfaces"
            )
            self.excluded_entities[0].mesh.export_stl(excluded_mesh_path)


def transform_Ce(
    raw_cp: pd.DataFrame, geometry_df: pd.DataFrame, _geometry: LnasGeometry
) -> pd.DataFrame:
    """Transforms pressure coefficient into shape coefficient

    Args:
        raw_cp (pd.DataFrame): Body pressure coefficient data
        geometry_df (pd.DataFrame): Dataframe with geometric properties and triangle indexing
        _geometry (LnasGeometry): Unused parameter to match function signature

    Returns:
        pd.DataFrame: Shape coefficient dataframe
    """
    cp_data = pd.merge(raw_cp, geometry_df, on="point_idx", how="inner")
    cp_data["f/q"] = cp_data["cp"] * cp_data["area"]

    Ce_data = (
        cp_data.groupby(["region_idx", "time_step"])  # type: ignore
        .agg(
            total_area=pd.NamedAgg(column="area", aggfunc="sum"),
            total_force=pd.NamedAgg(column="f/q", aggfunc="sum"),
        )
        .reset_index()
    )

    Ce_data["Ce"] = Ce_data["total_force"] / Ce_data["total_area"]
    Ce_data.drop(columns=["total_area", "total_force"], inplace=True)

    return Ce_data


def process_surfaces(
    geometry_dict: dict[str, GeometryData], cfg: CeConfig, ce_stats: pd.DataFrame
) -> list[ProcessedEntity]:
    """Generates a Processed surface for each of the body's surfaces

    Args:
        geometry_dict (dict[str, GeometryData]): Geometry data dictionary, keyed by surface label
        cfg (CeConfig): Shape coefficient configuration
        ce_stats (pd.DataFrame): Statistical values for each region of each surface

    Returns:
        list[ProcessedEntity]: List of processed surface. One for each of the values inside geometry_dict
    """
    processed_surfaces: list[ProcessedEntity] = []

    for sfc_lbl, geom_data in geometry_dict.items():
        regions_mesh, regions_mesh_triangles_indexing = generate_regions_mesh(
            geom_data=geom_data, cfg=cfg
        )
        regions_mesh_triangles_indexing = np.core.defchararray.add(
            regions_mesh_triangles_indexing.astype(str), "-" + sfc_lbl
        )
        region_data_df = combine_stats_data_with_mesh(
            regions_mesh, regions_mesh_triangles_indexing, ce_stats
        )
        if (region_data_df.isnull().sum() != 0).any():
            logger.warning(
                "Region refinement is greater than data refinement. Resulted in NaN values"
            )

        polydata = create_polydata_for_cell_data(region_data_df, regions_mesh)

        processed_surfaces.append(ProcessedEntity(mesh=regions_mesh, polydata=polydata))

    return processed_surfaces


def get_surface_dict(cfg: CeConfig, mesh: LnasFormat) -> dict[str, list[str]]:
    """Generates a dictionary with surface names keyed by the surface or set name

    Args:
        cfg (CeConfig): Shape coefficient configuration
        mesh (LnasFormat): Input mesh

    Returns:
        dict[str, list[str]]: Surface definition dictionary
    """
    sfc_dict = {set_lbl: sfc_list for set_lbl, sfc_list in cfg.sets.items()}
    sfc_dict |= {sfc: [sfc] for sfc in mesh.surfaces.keys() if sfc not in cfg.surfaces_in_sets}

    return sfc_dict


def get_region_definition_dataframe(geom_dict: dict[str, GeometryData]) -> pd.DataFrame:
    """Creates a dataframe with the resulting region index and its bounds (x_min, x_max, y_min, y_max, z_min, z_max)

    Args:
        geom_dict (dict[str, GeometryData]): Geometry data dictionary

    Returns:
        pd.DataFrame: Region definition dataframe
    """
    dfs = []
    for sfc_id, geom_data in geom_dict.items():
        df = pd.DataFrame()
        df = geom_data.zoning_to_use.get_regions_df()
        df["region_idx"] = df["region_idx"].astype(str) + f"-{sfc_id}"
        dfs.append(df)

    return pd.concat(dfs)


def process_Ce(
    mesh: LnasFormat,
    cfg: CeConfig,
    cp_path: pathlib.Path,
    time_scale_factor: float,
) -> CeOutput:
    """Executes the shape coefficient processing routine

    Args:
        mesh (LnasFormat): Input mesh
        cfg (CeConfig): Shape coefficient configuration
        cp_path (pathlib.Path): Path for pressure coefficient time series
        time_scale_factor (float): Factor for converting time scales from CST values

    Returns:
        CeOutput: Compiled outputs for shape coefficient use case
    """
    mesh_areas = mesh.geometry.areas
    mesh_normals = mesh.geometry.normals

    sfc_dict = get_surface_dict(cfg=cfg, mesh=mesh)

    logger.info("Getting geometry data...")
    geometry_dict = get_geometry_data(surface_dict=sfc_dict, cfg=cfg, mesh=mesh)

    logger.info("Tabulating geometry data...")
    geometry_df = tabulate_geometry_data(
        geom_dict=geometry_dict,
        mesh_areas=mesh_areas,
        mesh_normals=mesh_normals,
        transformation=cfg.transformation,
    )
    logger.info("Processing timesteps groups...")
    Ce_data = process_timestep_groups(
        data_path=cp_path,
        geometry_df=geometry_df,
        geometry=mesh.geometry,
        processing_function=transform_Ce,
    )
    Ce_data = convert_dataframe_into_matrix(
        Ce_data, column_data_label="region_idx", value_data_label="Ce"
    )

    logger.info("Calculating statistics...")
    Ce_stats = calculate_statistics(
        Ce_data, statistics_to_apply=cfg.statistics, time_scale_factor=time_scale_factor
    )

    logger.info("Processing surfaces...")
    processed_surfaces = process_surfaces(geometry_dict=geometry_dict, cfg=cfg, ce_stats=Ce_stats)
    logger.info("Processed surfaces!")

    excluded_sfc_list = [sfc for sfc in cfg.zoning.exclude if sfc in mesh.surfaces.keys()]  # type: ignore
    excluded_sfc_list += [
        sfc
        for set_lbl, sfc_set in cfg.sets.items()
        for sfc in sfc_set
        if set_lbl in cfg.zoning.exclude  # type: ignore
    ]
    if len(excluded_sfc_list) != 0:
        col = Ce_stats.columns
        excluded_surfaces = [
            get_excluded_entities(excluded_sfc_list=excluded_sfc_list, mesh=mesh, data_columns=col)
        ]
    else:
        excluded_surfaces = []

    ce_output = CeOutput(
        processed_entities=processed_surfaces,
        excluded_entities=excluded_surfaces,
        data_df=Ce_data,
        stats_df=Ce_stats,
        region_indexing_df=geometry_df[["region_idx", "point_idx"]],
        region_definition_df=get_region_definition_dataframe(geometry_dict),
    )

    return ce_output
