import pathlib
from dataclasses import dataclass

import numpy as np
import pandas as pd
from lnas import LnasFormat, LnasGeometry
from vtk import vtkPolyData

from cfdmod.api.vtk.write_vtk import create_polydata_for_cell_data, merge_polydata, write_polydata
from cfdmod.logger import logger
from cfdmod.use_cases.pressure.chunking import process_timestep_groups
from cfdmod.use_cases.pressure.extreme_values import ExtremeValuesParameters
from cfdmod.use_cases.pressure.geometry import (
    GeometryData,
    ProcessedEntity,
    get_excluded_entities,
    tabulate_geometry_data,
)
from cfdmod.use_cases.pressure.path_manager import CePathManager
from cfdmod.use_cases.pressure.shape.Ce_config import CeConfig
from cfdmod.use_cases.pressure.shape.Ce_geom import generate_regions_mesh, get_geometry_data
from cfdmod.use_cases.pressure.zoning.processing import (
    calculate_statistics,
    combine_stats_data_with_mesh,
)
from cfdmod.utils import create_folders_for_file


@dataclass
class CeOutputs:
    processed_surfaces: list[ProcessedEntity]
    excluded_surfaces: list[ProcessedEntity]
    Ce_data: pd.DataFrame
    Ce_stats: pd.DataFrame
    Ce_regions: pd.DataFrame

    def save_outputs(self, file_lbl: str, cfg_label: str, path_manager: CePathManager):
        # Output 1: Ce_regions
        regions_path = path_manager.get_regions_df_path(file_lbl, cfg_label)
        create_folders_for_file(regions_path)
        self.Ce_regions.to_hdf(path_or_buf=regions_path, key="Regions", mode="w", index=False)

        # Output 2: Ce(t)
        timeseries_path = path_manager.get_timeseries_df_path(file_lbl, cfg_label)
        self.Ce_data.to_hdf(path_or_buf=timeseries_path, key="Ce_t", mode="w", index=False)

        # Output 3: Ce_stats
        stats_path = path_manager.get_stats_df_path(file_lbl, cfg_label)
        self.Ce_stats.to_hdf(path_or_buf=stats_path, key="Ce_stats", mode="w", index=False)

        # Output 4: Regions Mesh
        mesh_path = path_manager.get_surface_path(file_lbl, cfg_label)
        regions_mesh = self.processed_surfaces[0].mesh.copy()
        regions_mesh.join([sfc.mesh.copy() for sfc in self.processed_surfaces[1:]])
        regions_mesh.export_stl(mesh_path)

        # Output 4 (Optional): Excluded Mesh
        if len(self.excluded_surfaces) != 0:
            excluded_mesh_path = path_manager.get_surface_path("excluded_surfaces", cfg_label)
            self.excluded_surfaces[0].mesh.export_stl(excluded_mesh_path)

        # Output 5: VTK polydata
        all_surfaces = self.processed_surfaces + self.excluded_surfaces
        merged_polydata = merge_polydata([surface_data.polydata for surface_data in all_surfaces])
        write_polydata(path_manager.get_vtp_path(file_lbl, cfg_label), merged_polydata)


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
        # cp_data.groupby(["region_idx", "sfc_idx", "time_step"])  # type: ignore
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


def combine_mesh_with_stats(
    regions_mesh: LnasGeometry,
    regions_mesh_triangles_indexing: np.ndarray,
    ce_stats: pd.DataFrame,
) -> vtkPolyData:
    """Generates a polydata combining the region mesh with Ce statistics values

    Args:
        regions_mesh (LnasGeometry): Region mesh
        regions_mesh_triangles_indexing (np.ndarray): Region mesh triangles indexing array
        ce_stats (pd.DataFrame): Dataframe with region statistics

    Returns:
        vtkPolyData: Combined polydata
    """
    region_data_df = combine_stats_data_with_mesh(
        regions_mesh, regions_mesh_triangles_indexing, ce_stats
    )
    if (region_data_df.isnull().sum() != 0).any():
        logger.warning("Region refinement is greater than data refinement. Resulted in NaN values")

    polydata = create_polydata_for_cell_data(region_data_df, regions_mesh)

    return polydata


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
        region_mesh, regions_mesh_triangles_indexing = generate_regions_mesh(
            geom_data=geom_data, cfg=cfg
        )
        regions_mesh_triangles_indexing = np.core.defchararray.add(
            regions_mesh_triangles_indexing.astype(str), "-" + sfc_lbl
        )
        polydata = combine_mesh_with_stats(
            regions_mesh=region_mesh,
            regions_mesh_triangles_indexing=regions_mesh_triangles_indexing,
            ce_stats=ce_stats,
        )

        processed_surfaces.append(ProcessedEntity(mesh=region_mesh, polydata=polydata))

    return processed_surfaces


def process_Ce(
    mesh: LnasFormat,
    cfg: CeConfig,
    cp_path: pathlib.Path,
    extreme_params: ExtremeValuesParameters | None,
) -> CeOutputs:
    """Executes the shape coefficient processing routine

    Args:
        mesh (LnasFormat): Input mesh
        cfg (CeConfig): Shape coefficient configuration
        cp_path (pathlib.Path): Path for pressure coefficient time series
        extreme_params (ExtremeValuesParameters | None): Optional parameters for extreme values analysis
        path_manager (CePathManager): Path manager
        cfg_label (str): Label of the current shape coefficient configuration

    Returns:
        CeOutputs: Compiled outputs for shape coefficient use case
    """
    mesh_areas = mesh.geometry.areas
    mesh_normals = mesh.geometry.normals

    sfc_dict = {set_lbl: sfc_list for set_lbl, sfc_list in cfg.sets.items()}
    sfc_dict |= {sfc: [sfc] for sfc in mesh.surfaces.keys() if sfc not in cfg.surfaces_in_sets}

    geometry_dict = get_geometry_data(surface_dict=sfc_dict, cfg=cfg, mesh=mesh)
    geometry_df = tabulate_geometry_data(
        geom_dict=geometry_dict,
        mesh_areas=mesh_areas,
        mesh_normals=mesh_normals,
        transformation=cfg.transformation,
    )
    Ce_data = process_timestep_groups(
        data_path=cp_path,
        geometry_df=geometry_df,
        geometry=mesh.geometry,
        processing_function=transform_Ce,
    )
    Ce_stats = calculate_statistics(
        Ce_data,
        statistics_to_apply=cfg.statistics,
        variables=["Ce"],
        group_by_key="region_idx",
        extreme_params=extreme_params,
    )

    processed_surfaces = process_surfaces(geometry_dict=geometry_dict, cfg=cfg, ce_stats=Ce_stats)

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

    ce_output = CeOutputs(
        processed_surfaces=processed_surfaces,
        excluded_surfaces=excluded_surfaces,
        Ce_data=Ce_data,
        Ce_stats=Ce_stats,
        Ce_regions=geometry_df,
    )

    return ce_output
