import numpy as np
import pandas as pd
from nassu.lnas import LagrangianGeometry

from cfdmod.use_cases.pressure.statistics import Statistics


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
    triangles_normals = surface_mesh._cross_prod()
    triangles_areas = np.linalg.norm(triangles_normals, axis=1)

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
        statistics_to_apply (Statistics): List of statistical functions to apply

    Returns:
        pd.DataFrame: Statistics for shape coefficient
    """
    group_by_point = region_data.groupby("region_idx")

    statistics_data = pd.DataFrame({"region_idx": region_data["region_idx"].unique()})

    # BUG: NaN values
    # if "avg" in statistics_to_apply:
    #     statistics_data["Ce_avg"] = group_by_point["Ce"].mean()
    # if "min" in statistics_to_apply:
    #     statistics_data["Ce_min"] = group_by_point["Ce"].min()
    # if "max" in statistics_to_apply:
    #     statistics_data["Ce_max"] = group_by_point["Ce"].max()
    # if "std" in statistics_to_apply:
    #     statistics_data["Ce_rms"] = group_by_point["Ce"].std()

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
