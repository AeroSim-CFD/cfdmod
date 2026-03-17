"""Geometry utilities for the pressure module.

Contains GeometryData/ProcessedEntity dataclasses, tabulation helpers,
and Ce surface zoning geometry (formerly Ce_geom.py).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from lnas import LnasFormat, LnasGeometry

from cfdmod.io.geometry.region_meshing import create_regions_mesh
from cfdmod.io.geometry.transformation_config import TransformationConfig
from cfdmod.logger import logger
from cfdmod.pressure.parameters import BodyConfig, CeConfig, MomentBodyConfig, ZoningModel


# ---------------------------------------------------------------------------
# Shared geometry helpers
# ---------------------------------------------------------------------------


@dataclass
class GeometryData:
    mesh: LnasGeometry
    zoning_to_use: ZoningModel
    triangles_idxs: np.ndarray


@dataclass
class ProcessedEntity:
    mesh: LnasGeometry


def get_region_definition_dataframe(geom_dict: dict[str, GeometryData]) -> pd.DataFrame:
    """Creates a dataframe with region index and bounds.

    Args:
        geom_dict (dict[str, GeometryData]): Geometry data dictionary

    Returns:
        pd.DataFrame: Region definition dataframe
    """
    dfs = []
    for sfc_id, geom_data in geom_dict.items():
        df = geom_data.zoning_to_use.get_regions_df()
        df["region_idx"] = df["region_idx"].astype(str) + f"-{sfc_id}"
        dfs.append(df)
    return pd.concat(dfs)


def get_indexing_mask(mesh: LnasGeometry, df_regions: pd.DataFrame) -> np.ndarray:
    """Index each triangle in the mesh into the respective region.

    Args:
        mesh (LnasGeometry): Mesh with triangles to index
        df_regions (pd.DataFrame): Dataframe describing zoning intervals
            (x_min, x_max, y_min, y_max, z_min, z_max, region_idx)

    Returns:
        np.ndarray: Triangle region indexing array
    """
    triangles = mesh.triangle_vertices
    centroids = np.mean(triangles, axis=1)
    triangles_region = np.full((triangles.shape[0],), -1, dtype=np.int32)

    for _, region in df_regions.iterrows():
        ll = np.array([region["x_min"], region["y_min"], region["z_min"]])
        ur = np.array([region["x_max"], region["y_max"], region["z_max"]])
        in_idx = np.all(
            np.logical_and(centroids >= ll, centroids < ur),
            axis=1,
        )
        triangles_region[in_idx] = region["region_idx"]

    return triangles_region


def get_region_indexing(
    geom_data: GeometryData,
    transformation: TransformationConfig,
) -> np.ndarray:
    """Index each triangle from the geometry after applying transformation.

    Args:
        geom_data (GeometryData): Geometry data
        transformation (TransformationConfig): Transformation configuration

    Returns:
        np.ndarray: Triangle indexing array
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
    """Convert a dictionary of GeometryData into a DataFrame with geometric properties.

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
        df["region_idx"] = np.char.add(region_idx_per_tri.astype(str), f"-{sfc_id}")
        df["point_idx"] = geom_data.triangles_idxs
        df["area"] = mesh_areas[geom_data.triangles_idxs].copy()
        df["n_x"] = mesh_normals[geom_data.triangles_idxs, 0].copy()
        df["n_y"] = mesh_normals[geom_data.triangles_idxs, 1].copy()
        df["n_z"] = mesh_normals[geom_data.triangles_idxs, 2].copy()
        dfs.append(df)
    return pd.concat(dfs)


def get_geometry_data(
    body_cfg: BodyConfig | MomentBodyConfig, sfc_list: list[str], mesh: LnasFormat
) -> GeometryData:
    """Build a GeometryData from the mesh and body configuration.

    Args:
        body_cfg (BodyConfig | MomentBodyConfig): Body configuration
        sfc_list (list[str]): List of surfaces composing the body
        mesh (LnasFormat): Input mesh

    Returns:
        GeometryData: Filtered GeometryData
    """
    sfcs = sfc_list if len(sfc_list) != 0 else list(mesh.surfaces.keys())
    geom, geometry_idx = mesh.geometry_from_list_surfaces(surfaces_names=sfcs)
    return GeometryData(mesh=geom, zoning_to_use=body_cfg.sub_bodies, triangles_idxs=geometry_idx)


def combine_stats_data_with_mesh(
    mesh: LnasGeometry,
    region_idx_array: np.ndarray,
    data_stats: pd.DataFrame,
) -> pd.DataFrame:
    """Combine compiled statistical data with surface meshing by indexing regions.

    Args:
        mesh (LnasGeometry): LNAS mesh to be combined
        region_idx_array (np.ndarray): Triangles indexing by region
        data_stats (pd.DataFrame): Compiled statistics data

    Returns:
        pd.DataFrame: Dataframe with region statistics indexed by mesh triangles
    """
    combined_df = pd.DataFrame()
    combined_df["point_idx"] = np.arange(len(mesh.triangle_vertices))
    combined_df["region_idx"] = region_idx_array
    combined_df = pd.merge(combined_df, data_stats, on="region_idx", how="left")
    combined_df.drop(columns=["region_idx"], inplace=True)
    return combined_df


# ---------------------------------------------------------------------------
# Ce surface geometry (formerly Ce_geom.py)
# ---------------------------------------------------------------------------


def _get_surface_zoning(mesh: LnasGeometry, sfc: str, config: CeConfig) -> ZoningModel:
    """Get the surface zoning configuration.

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
        zoning = [
            cfg for cfg in config.zoning.exceptions.values() if sfc in cfg.surfaces  # type: ignore
        ][0]
    else:
        zoning = config.zoning.global_zoning  # type: ignore
        if len(np.unique(np.round(mesh.normals, decimals=2), axis=0)) == 1:
            ignore_axis = np.where(
                np.abs(mesh.normals[0]) == np.abs(mesh.normals[0]).max()
            )[0][0]
            zoning = zoning.ignore_axis(ignore_axis)

    return zoning.offset_limits(0.1)


def get_ce_geometry_data(
    surface_dict: dict[str, list[str]], cfg: CeConfig, mesh: LnasFormat
) -> dict[str, GeometryData]:
    """Get surfaces geometry data from mesh for Ce (shape coefficient).

    Args:
        surface_dict (dict[str, list[str]]): Surface list keyed by label
        cfg (CeConfig): Post processing configuration
        mesh (LnasFormat): LNAS mesh

    Returns:
        dict[str, GeometryData]: Geometry data keyed by surface label
    """
    geom_dict: dict[str, GeometryData] = {}
    for sfc_lbl, sfc_list in surface_dict.items():
        if sfc_lbl in cfg.zoning.exclude:  # type: ignore
            logger.debug(f"Surface {sfc_lbl} ignored!")
            continue
        surface_geom, sfc_triangles_idxs = mesh.geometry_from_list_surfaces(
            surfaces_names=sfc_list
        )
        zoning_to_use = _get_surface_zoning(mesh=surface_geom, sfc=sfc_lbl, config=cfg)
        geom_dict[sfc_lbl] = GeometryData(
            mesh=surface_geom,
            zoning_to_use=zoning_to_use,
            triangles_idxs=sfc_triangles_idxs,
        )
    return geom_dict


def generate_regions_mesh(
    geom_data: GeometryData, cfg: CeConfig
) -> tuple[LnasGeometry, np.ndarray]:
    """Generate a new mesh intersecting the input mesh with the regions definition.

    Args:
        geom_data (GeometryData): Geometry data with surface mesh and regions
        cfg (CeConfig): Shape coefficient configuration

    Returns:
        tuple[LnasGeometry, np.ndarray]: Region mesh and triangle indexing
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
    regions_mesh_triangles_indexing = get_indexing_mask(
        mesh=regions_mesh, df_regions=df_regions
    )
    regions_mesh.apply_transformation(
        cfg.transformation.get_geometry_transformation(), invert_transf=True
    )
    return regions_mesh, regions_mesh_triangles_indexing
