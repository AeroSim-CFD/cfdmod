import numpy as np
import pandas as pd
from nassu.lnas import LagrangianGeometry

from cfdmod.use_cases.pressure.shape.Ce_config import CeConfig
from cfdmod.use_cases.pressure.shape.region_meshing import get_mesh_bounds
from cfdmod.use_cases.pressure.shape.zoning_config import ZoningModel
from cfdmod.use_cases.pressure.statistics import Statistics


def get_surface_zoning(mesh: LagrangianGeometry, sfc: str, config: CeConfig) -> ZoningModel:
    """Get the surface respective zoning configuration

    Args:
        mesh (LagrangianGeometry): Surface LNAS mesh
        sfc (str): Surface label
        config (CeConfig): Post process configuration

    Returns:
        ZoningModel: Zoning configuration
    """
    if sfc in config.zoning.no_zoning:
        bounds = get_mesh_bounds(mesh)
        zoning = ZoningModel(
            x_intervals=[bounds[0][0], bounds[0][1]],
            y_intervals=[bounds[1][0], bounds[1][1]],
            z_intervals=[bounds[2][0], bounds[2][1]],
        )
    elif sfc in config.zoning.surfaces_in_exception:
        zoning = [cfg for cfg in config.zoning.exceptions.values() if sfc in cfg.surfaces][0]
    else:
        zoning = config.zoning.global_zoning

    if len(np.unique(np.round(mesh.normals, decimals=2), axis=0)) == 1:
        axis_to_ignore = np.where(np.abs(mesh.normals[0]) == np.abs(mesh.normals[0]).max())[0][0]
        zoning = zoning.ignore_axis(axis_to_ignore)

    return zoning.offset_limits(0.1)


def combine_region_data_with_mesh(
    regions_mesh: LagrangianGeometry,
    regions_mesh_triangles_region: np.ndarray,
    surface_ce_stats: pd.DataFrame,
) -> pd.DataFrame:
    """Combine compiled region data with surface meshing by indexing regions

    Args:
        regions_mesh (LagrangianGeometry): Generated mesh with region separation
        regions_mesh_triangles_region (np.ndarray): Region mesh triangles indexing by region
        surface_ce_stats (pd.DataFrame): Compiled region statistics data

    Returns:
        pd.DataFrame: Region mesh dataframe with region statistics
    """
    region_data_df = pd.DataFrame()
    region_data_df["point_idx"] = np.arange(len(regions_mesh.triangle_vertices))
    region_data_df["region_idx"] = regions_mesh_triangles_region
    region_data_df = pd.merge(region_data_df, surface_ce_stats, on="region_idx", how="left")
    region_data_df.drop(columns=["region_idx"], inplace=True)

    return region_data_df


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


def calculate_statistics(
    region_data: pd.DataFrame, statistics_to_apply: list[Statistics]
) -> pd.DataFrame:
    """Calculates statistics for pressure coefficient of a body data

    Args:
        region_data (pd.DataFrame): Dataframe of the region data shape coefficients
        statistics_to_apply (list[Statistics]): List of statistical functions to apply

    Returns:
        pd.DataFrame: Statistics for shape coefficient
    """
    group_by_point = region_data.groupby("region_idx")

    statistics_data = pd.DataFrame({"region_idx": region_data["region_idx"].unique()})

    if "avg" in statistics_to_apply:
        average = group_by_point["Ce"].apply(lambda x: x.mean()).reset_index(name="mean")
        statistics_data["Ce_avg"] = average["mean"]
    if "min" in statistics_to_apply:
        minimum = group_by_point["Ce"].apply(lambda x: x.min()).reset_index(name="min")
        statistics_data["Ce_min"] = minimum["min"]
    if "max" in statistics_to_apply:
        maximum = group_by_point["Ce"].apply(lambda x: x.max()).reset_index(name="max")
        statistics_data["Ce_max"] = maximum["max"]
    if "std" in statistics_to_apply:
        rms = group_by_point["Ce"].apply(lambda x: x.std()).reset_index(name="std")
        statistics_data["Ce_rms"] = rms["std"]

    # Calculate skewness and kurtosis using apply
    if "skewness" in statistics_to_apply:
        skewness = group_by_point["Ce"].apply(lambda x: x.skew()).reset_index(name="skewness")
        statistics_data["Ce_skewness"] = skewness["skewness"]
    if "kurtosis" in statistics_to_apply:
        kurtosis = group_by_point["Ce"].apply(lambda x: x.kurt()).reset_index(name="kurtosis")
        statistics_data["Ce_kurtosis"] = kurtosis["kurtosis"]

    return statistics_data
