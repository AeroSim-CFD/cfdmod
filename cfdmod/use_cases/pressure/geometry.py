from dataclasses import dataclass

import numpy as np
import pandas as pd
from lnas import LnasFormat, LnasGeometry
from vtk import vtkPolyData

from cfdmod.api.geometry.transformation_config import TransformationConfig
from cfdmod.api.vtk.write_vtk import create_polydata_for_cell_data
from cfdmod.use_cases.pressure.shape.zoning_config import ZoningModel
from cfdmod.use_cases.pressure.zoning.body_config import BodyConfig
from cfdmod.use_cases.pressure.zoning.processing import get_indexing_mask


@dataclass
class GeometryData:
    mesh: LnasGeometry
    zoning_to_use: ZoningModel
    triangles_idxs: np.ndarray


@dataclass
class ProcessedEntity:
    mesh: LnasGeometry
    polydata: vtkPolyData


def get_excluded_entities(
    excluded_sfc_list: list[str], mesh: LnasFormat, data_columns: list[str]
) -> ProcessedEntity:
    """Generates a Processed entity for the excluded surfaces

    Args:
        excluded_sfc_list (list[str]): List of excluded surfaces
        mesh (LnasFormat): Original input mesh
        data_columns (list[str]): Name of the data columns to be spawned as NaN

    Returns:
        ProcessedEntity: Processed entity for excluded surfaces
    """
    excluded_sfcs, _ = mesh.geometry_from_list_surfaces(surfaces_names=excluded_sfc_list)
    columns = [col for col in data_columns if col not in ["point_idx", "region_idx"]]
    excluded_polydata = create_NaN_polydata(mesh=excluded_sfcs, column_labels=columns)

    return ProcessedEntity(mesh=excluded_sfcs, polydata=excluded_polydata)


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


def create_NaN_polydata(mesh: LnasGeometry, column_labels: list[str]) -> vtkPolyData:
    """Creates vtkPolyData from a given mesh and populate column labels with NaN values

    Args:
        mesh (LnasGeometry): Input LNAS mesh
        column_labels (list[str]): Column labels to populate with NaN values

    Returns:
        vtkPolyData: Polydata with the input mesh and NaN values
    """
    mock_df = pd.DataFrame(columns=column_labels)
    mock_df["point_idx"] = np.arange(0, mesh.triangles.shape[0])
    # All other columns will be NaN except for point_idx
    polydata = create_polydata_for_cell_data(data=mock_df, mesh=mesh)

    return polydata


def get_region_indexing(
    geom_data: GeometryData,
    transformation: TransformationConfig,
) -> np.ndarray:
    """Index each triangle from the geometry after applying transformation

    Args:
        geom_data (GeometryData): Geometry data
        transformation (TransformationConfig): Transformation configuration

    Returns:
        np.ndarray: Triangle indexing. Each triangle of the geometry has a corresponding region index
    """
    df_regions = geom_data.zoning_to_use.get_regions_df()

    transformed_geometry = geom_data.mesh.copy()
    transformed_geometry.apply_transformation(transformation.get_geometry_transformation())

    triangles_region_idx = get_indexing_mask(mesh=transformed_geometry, df_regions=df_regions)

    return triangles_region_idx


def tabulate_geometry_data(
    geom_dict: dict[str, GeometryData],
    mesh_areas: np.ndarray,
    mesh_normals: np.ndarray,
    transformation: TransformationConfig,
) -> pd.DataFrame:
    """Converts a dictionary of GeometryData into a DataFrame with geometric properties

    Args:
        geom_dict (dict[str, GeometryData]): Geometry data dictionary
        mesh_areas (np.ndarray): Parent mesh areas
        mesh_normals (np.ndarray): Parent mesh normals
        transformation (TransformationConfig): Transformation configuration

    Returns:
        pd.DataFrame: Geometry data tabulated into a DataFrame
    """
    dfs = []

    for sfc_id, geom_data in geom_dict.items():
        df = pd.DataFrame()
        region_idx_per_tri = get_region_indexing(
            geom_data=geom_data, transformation=transformation
        )
        df["region_idx"] = np.core.defchararray.add(region_idx_per_tri.astype(str), "-" + sfc_id)
        df["point_idx"] = geom_data.triangles_idxs
        df["area"] = mesh_areas[geom_data.triangles_idxs].copy()
        df["n_x"] = mesh_normals[geom_data.triangles_idxs, 0].copy()
        df["n_y"] = mesh_normals[geom_data.triangles_idxs, 1].copy()
        df["n_z"] = mesh_normals[geom_data.triangles_idxs, 2].copy()
        dfs.append(df)

    geometry_df = pd.concat(dfs)

    return geometry_df


def get_geometry_data(body_cfg: BodyConfig, sfc_list: list[str], mesh: LnasFormat) -> GeometryData:
    """Builds a GeometryData from the mesh and the configurations

    Args:
        body_cfg (BodyConfig): Body configuration with zoning parameters
        sfc_list (list[str]): List of surfaces that compose the body
        mesh (LnasFormat): Input mesh

    Returns:
        GeometryData: Filtered GeometryData
    """
    sfcs = sfc_list if len(sfc_list) != 0 else [k for k in mesh.surfaces.keys()]
    geom, geometry_idx = mesh.geometry_from_list_surfaces(surfaces_names=sfcs)

    return GeometryData(mesh=geom, zoning_to_use=body_cfg.sub_bodies, triangles_idxs=geometry_idx)
