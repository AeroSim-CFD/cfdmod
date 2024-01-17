import pathlib
from dataclasses import dataclass

import numpy as np
import pandas as pd
from lnas import LnasFormat, LnasGeometry
from vtk import vtkPolyData

from cfdmod.api.geometry.region_meshing import create_regions_mesh
from cfdmod.api.vtk.write_vtk import (
    create_polydata_for_cell_data,
    merge_polydata,
    vtkPolyData,
    write_polydata,
)
from cfdmod.logger import logger
from cfdmod.use_cases.pressure.chunking import process_timestep_groups
from cfdmod.use_cases.pressure.extreme_values import ExtremeValuesParameters
from cfdmod.use_cases.pressure.geometry import (
    GeometryData,
    combine_geometries,
    create_NaN_polydata,
    filter_geometry_from_list,
    get_excluded_surfaces,
    get_region_indexing,
)
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
class ProcessedSurface:
    regions_mesh: LnasGeometry
    polydata: vtkPolyData


@dataclass
class CeOutputs:
    processed_surfaces: list[ProcessedSurface]
    excluded_surfaces: list[ProcessedSurface]
    Ce_data: pd.DataFrame
    Ce_stats: pd.DataFrame
    Ce_regions: pd.DataFrame

    def save_outputs(self, mesh_name: str, cfg_label: str, path_manager: CePathManager):
        # Output 1: Ce_regions
        regions_path = path_manager.get_regions_df_path(mesh_name, cfg_label)
        create_folders_for_file(regions_path)
        self.Ce_regions.to_hdf(regions_path, key="Regions", mode="w", index=False)

        # Output 2: Ce(t)
        timeseries_path = path_manager.get_timeseries_df_path(mesh_name, cfg_label)
        self.Ce_data.to_hdf(timeseries_path, key="Ce_t", mode="w", index=False)

        # Output 3: Ce_stats
        stats_path = path_manager.get_stats_df_path(mesh_name, cfg_label)
        self.Ce_stats.to_hdf(stats_path, key="Ce_stats", mode="w", index=False)

        # Output 4: Regions Mesh
        mesh_path = path_manager.get_surface_path(mesh_name, cfg_label)
        regions_mesh = combine_geometries([sfc.regions_mesh for sfc in self.processed_surfaces])
        regions_mesh.export_stl(mesh_path)

        # Output 4 (Optional): Excluded Mesh
        if len(self.excluded_surfaces) != 0:
            excluded_mesh_path = path_manager.get_surface_path("excluded_surfaces", cfg_label)
            excluded_mesh = combine_geometries([s.regions_mesh for s in self.excluded_surfaces])
            excluded_mesh.export_stl(excluded_mesh_path)

        # Output 5: VTK polydata
        all_surfaces = self.processed_surfaces + self.excluded_surfaces
        merged_polydata = merge_polydata([surface_data.polydata for surface_data in all_surfaces])
        write_polydata(path_manager.get_vtp_path(mesh_name, cfg_label), merged_polydata)


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
    raw_surfaces: dict[str, GeometryData] = {}
    for sfc_lbl, sfc_list in surface_dict.items():
        if sfc_lbl in cfg.zoning.exclude:  # type: ignore (already validated in class)
            logger.info(f"Surface {sfc_lbl} ignored!")  # Ignore surface
            continue
        surface_geom, sfc_triangles_idxs = filter_geometry_from_list(mesh=mesh, sfc_list=sfc_list)
        zoning_to_use = get_surface_zoning(mesh=surface_geom, sfc=sfc_lbl, config=cfg)

        raw_surface = GeometryData(
            mesh=surface_geom,
            zoning_to_use=zoning_to_use,
            triangles_idxs=sfc_triangles_idxs,
        )
        raw_surfaces[sfc_lbl] = raw_surface
    return raw_surfaces


def tabulate_geometry_data(
    geom_dict: dict[str, GeometryData],
    mesh_areas: np.ndarray,
    mesh_normals: np.ndarray,
    cfg: CeConfig,
) -> pd.DataFrame:
    """Converts a dictionary of GeometryData into a DataFrame with geometric properties

    Args:
        geom_dict (dict[str, GeometryData]): Geometry data dictionary
        mesh_areas (np.ndarray): Parent mesh areas
        mesh_normals (np.ndarray): Parent mesh normals
        cfg (CeConfig): Shape coefficient configuration

    Returns:
        pd.DataFrame: Geometry data tabulated into a DataFrame
    """
    dfs = []

    for sfc_id, geom_data in geom_dict.items():
        df = pd.DataFrame()
        region_idx_per_tri = get_region_indexing(
            geom_data=geom_data, transformation=cfg.transformation
        )
        df["region_idx"] = np.core.defchararray.add(region_idx_per_tri.astype(str), "-" + sfc_id)
        # df["region_idx"] = np.array(map(lambda x: (x, sfc_id), region_idx_per_tri))
        # Compose keys
        # df["region_idx"] = region_idx_per_tri
        df["point_idx"] = geom_data.triangles_idxs
        df["area"] = mesh_areas[geom_data.triangles_idxs].copy()
        df["n_x"] = mesh_normals[geom_data.triangles_idxs, 0].copy()
        df["n_y"] = mesh_normals[geom_data.triangles_idxs, 1].copy()
        df["n_z"] = mesh_normals[geom_data.triangles_idxs, 2].copy()
        # df["sfc_idx"] = sfc_id
        dfs.append(df)

    geometry_df = pd.concat(dfs)

    return geometry_df


def transform_Ce(
    raw_cp: pd.DataFrame,
    geometry_df: pd.DataFrame,
) -> pd.DataFrame:
    """Transforms pressure coefficient into shape coefficient

    Args:
        raw_cp (pd.DataFrame): Body pressure coefficient data
        geometry_df (pd.DataFrame): Dataframe with geometric properties and triangle indexing

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
    # Ce_data["region_idx"] = Ce_data["region_idx"].astype(str) + "-" + Ce_data["sfc_idx"]
    Ce_data.drop(columns=["total_area", "total_force"], inplace=True)
    # Ce_data.drop(columns=["total_area", "total_force", "sfc_idx"], inplace=True)

    return Ce_data


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
) -> list[ProcessedSurface]:
    """Generates a Processed surface for each of the body's surfaces

    Args:
        geometry_dict (dict[str, GeometryData]): Geometry data dictionary, keyed by surface label
        cfg (CeConfig): Shape coefficient configuration
        ce_stats (pd.DataFrame): Statistical values for each region of each surface

    Returns:
        list[ProcessedSurface]: List of processed surface. One for each of the values inside geometry_dict
    """
    processed_surfaces: list[ProcessedSurface] = []

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

        processed_surfaces.append(ProcessedSurface(regions_mesh=region_mesh, polydata=polydata))

    return processed_surfaces


def process_Ce(
    mesh: LnasFormat,
    cfg: CeConfig,
    cp_path: pathlib.Path,
    extreme_params: ExtremeValuesParameters,
) -> CeOutputs:
    """Executes the shape coefficient processing routine

    Args:
        mesh (LnasFormat): Input mesh
        cfg (CeConfig): Shape coefficient configuration
        cp_path (pathlib.Path): Path for pressure coefficient time series
        extreme_params (ExtremeValuesParameters): Parameters for extreme values analysis
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
        cfg=cfg,
    )
    Ce_data = process_timestep_groups(
        data_path=cp_path, geometry_df=geometry_df, processing_function=transform_Ce
    )
    Ce_stats = calculate_statistics(
        Ce_data,
        statistics_to_apply=cfg.statistics,
        variables=["Ce"],
        group_by_key="region_idx",
        extreme_params=extreme_params,
    )

    processed_surfaces = process_surfaces(geometry_dict=geometry_dict, cfg=cfg, ce_stats=Ce_stats)
    excluded_surfaces = get_excluded_surfaces_data(
        cfg=cfg, mesh=mesh, data_columns=Ce_stats.columns
    )

    ce_output = CeOutputs(
        processed_surfaces=processed_surfaces,
        excluded_surfaces=excluded_surfaces,
        Ce_data=Ce_data,
        Ce_stats=Ce_stats,
        Ce_regions=geometry_df,
    )

    return ce_output


def get_excluded_surfaces_data(
    cfg: CeConfig, mesh: LnasFormat, data_columns: list[str]
) -> list[ProcessedSurface]:
    """Generates a Processed surface for the excluded surfaces

    Args:
        cfg (CeConfig): Shape coefficient configuration
        mesh (LnasFormat): Original input mesh
        data_columns (list[str]): Name of the data columns to be spawned as NaN

    Returns:
        list[ProcessedSurface]: List of processed excluded surfaces. Empty if there is not a excluded surface
    """
    sfc_list = [sfc for sfc in cfg.zoning.exclude if sfc in mesh.surfaces.keys()]  # type: ignore
    sfc_list += [
        sfc
        for set_lbl, sfc_set in cfg.sets.items()
        for sfc in sfc_set
        if set_lbl in cfg.zoning.exclude  # type: ignore
    ]

    if len(sfc_list) != 0:
        excluded_sfcs = get_excluded_surfaces(mesh=mesh, sfc_list=sfc_list)
        columns = [col for col in data_columns if col not in ["point_idx", "region_idx"]]
        excluded_polydata = create_NaN_polydata(mesh=excluded_sfcs, column_labels=columns)
        return [ProcessedSurface(regions_mesh=excluded_sfcs, polydata=excluded_polydata)]
    else:
        return []
