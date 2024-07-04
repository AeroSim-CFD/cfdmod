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
    get_region_definition_dataframe,
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
    time_normalized = raw_cp["time_normalized"].copy()
    cols_points = [c for c in raw_cp.columns if c != "time_normalized"]
    id_points = np.array([int(c) for c in cols_points])

    points_selection = geometry_df.sort_values(by="point_idx")["point_idx"].to_numpy()
    face_area = geometry_df["area"].to_numpy()

    mask_valid_points = np.isin(id_points, points_selection)
    id_points_selected = id_points[mask_valid_points]
    cp_matrix = raw_cp[cols_points].copy().to_numpy()[:, mask_valid_points]

    regions_list = geometry_df["region_idx"].unique()

    f_q_matrix = cp_matrix * face_area

    list_of_ce_region = []
    for region in regions_list:
        points_of_region = geometry_df[geometry_df["region_idx"] == region]["point_idx"].to_numpy()
        mask_points_of_region = np.isin(id_points_selected, points_of_region)

        ce_region = pd.DataFrame(
            {
                "time_normalized": time_normalized,
                "f/q": np.sum(f_q_matrix[:, mask_points_of_region], axis=1),
                "area": np.sum(face_area[mask_points_of_region]),
                "region_idx": region,
            }
        )
        list_of_ce_region.append(ce_region)

    ce_full = pd.concat(list_of_ce_region)
    del list_of_ce_region

    Ce_data = (
        ce_full.groupby(["region_idx", "time_normalized"])  # type: ignore
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
) -> tuple[list[ProcessedEntity], pd.DataFrame]:
    """Generates a Processed surface for each of the body's surfaces

    Args:
        geometry_dict (dict[str, GeometryData]): Geometry data dictionary, keyed by surface label
        cfg (CeConfig): Shape coefficient configuration
        ce_stats (pd.DataFrame): Statistical values for each region of each surface

    Returns:
        tuple[list[ProcessedEntity], pd.DataFrame]: Tuple with a list of processed surface, one for each of the values inside geometry_dict;
        and region indexing dataframe (point_idx, region_idx)
    """
    processed_surfaces: list[ProcessedEntity] = []

    region_indexing_dfs = []
    last_index_recorded = 0
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
        indexing_df = pd.DataFrame(
            {
                "point_idx" : np.arange(len(regions_mesh.triangle_vertices)) + last_index_recorded, 
                "region_idx": regions_mesh_triangles_indexing
            }
        )
        region_indexing_dfs.append(indexing_df)
        last_index_recorded += len(regions_mesh.triangle_vertices)
        polydata = create_polydata_for_cell_data(region_data_df, regions_mesh)

        processed_surfaces.append(ProcessedEntity(mesh=regions_mesh, polydata=polydata))

    return processed_surfaces, pd.concat(region_indexing_dfs)


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


def process_Ce(
    mesh: LnasFormat,
    cfg: CeConfig,
    cp_path: pathlib.Path,
) -> CeOutput:
    """Executes the shape coefficient processing routine

    Args:
        mesh (LnasFormat): Input mesh
        cfg (CeConfig): Shape coefficient configuration
        cp_path (pathlib.Path): Path for pressure coefficient time series

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
        Ce_data,
        row_data_label="time_normalized",
        column_data_label="region_idx",
        value_data_label="Ce",
    )

    logger.info("Calculating statistics...")
    Ce_stats = calculate_statistics(Ce_data, statistics_to_apply=cfg.statistics)

    logger.info("Processing surfaces...")
    processed_surfaces, regions_indexing_df = process_surfaces(geometry_dict=geometry_dict, cfg=cfg, ce_stats=Ce_stats)
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
        excluded_entities = [
            get_excluded_entities(excluded_sfc_list=excluded_sfc_list, mesh=mesh, data_columns=col)
        ]
    else:
        excluded_entities = []

    ce_output = CeOutput(
        processed_entities=processed_surfaces,
        excluded_entities=excluded_entities,
        data_df=Ce_data,
        stats_df=Ce_stats,
        region_indexing_df=regions_indexing_df[["region_idx", "point_idx"]],
        region_definition_df=get_region_definition_dataframe(geometry_dict),
    )

    return ce_output
