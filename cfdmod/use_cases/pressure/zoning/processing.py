from typing import Literal, Optional

import numpy as np
import pandas as pd
from lnas import LnasGeometry

from cfdmod.use_cases.pressure.extreme_values import (
    gumbel_extreme_values,
    moving_average_extreme_values,
    peak_extreme_values,
)
from cfdmod.use_cases.pressure.statistics import (
    BasicStatisticModel,
    ParameterizedStatisticModel,
    StatisticsParamsModel,
)

AxisDirections = Literal["x", "y", "z"]


def get_indexing_mask(mesh: LnasGeometry, df_regions: pd.DataFrame) -> np.ndarray:
    """Index each triangle in the mesh in the respective region

    Args:
        mesh (LnasGeometry): Mesh with triangles to index
        df_regions (pd.DataFrame): Dataframe describing the zoning intervals (x_min, x_max, y_min, y_max, z_min, z_max, region_idx)

    Returns:
        np.ndarray: Triangles zoning indexing array
    """
    triangles = mesh.triangle_vertices
    centroids = np.mean(triangles, axis=1)

    triangles_region = np.full((triangles.shape[0],), -1, dtype=np.int32)

    for _, region in df_regions.iterrows():
        ll = np.array([region["x_min"], region["y_min"], region["z_min"]])  # lower-left
        ur = np.array([region["x_max"], region["y_max"], region["z_max"]])  # upper-right

        in_idx = np.all(
            np.logical_and(
                centroids >= ll,
                centroids < ur,
            ),
            axis=1,
        )
        triangles_region[in_idx] = region["region_idx"]

    return triangles_region


def extreme_values_analysis(
    params: StatisticsParamsModel,
    data_df: pd.DataFrame,
    timestep_arr: np.ndarray,
) -> pd.DataFrame:
    """Perform extreme values analysis to a dataframe

    Args:
        params (StatisticsParamsModel): Extreme values parameters
        data_df (pd.DataFrame): Input dataframe in matrix form
        timestep_arr (np.ndarray, optional): Time step array for Gumbel method.

    Returns:
        pd.DataFrame: _description_
    """
    stat_df = pd.DataFrame()
    if params.method_type == "Absolute":
        stat_df = data_df.apply(lambda x: (x.min(), x.max()))
    elif params.method_type == "Gumbel":
        stat_df = data_df.apply(
            lambda x: gumbel_extreme_values(
                params=params,
                timestep_arr=timestep_arr,
                hist_series=x,
            )
        )
    elif params.method_type == "Peak":
        stat_df = data_df.apply(
            lambda x: peak_extreme_values(
                params=params,
                hist_series=x,
            )
        )
    elif params.method_type == "Moving Average":
        stat_df = data_df.apply(
            lambda x: moving_average_extreme_values(
                params=params,
                hist_series=x,
            )
        )
    return stat_df


def calculate_extreme_values(
    extreme_statistics: list[ParameterizedStatisticModel],
    timestep_arr: np.ndarray,
    data_df: pd.DataFrame,
) -> dict[str, pd.DataFrame]:
    """Calculates extreme values from historical data

    Args:
        extreme_statistics (list[ParameterizedStatisticModel]): List of min and max statistical model parameters
        timestep_arr (np.ndarray): Time step array for Gumbel and Moving Average methods
        data_df (pd.DataFrame): Point hist series dataframe

    Returns:
        dict[str, pd.DataFrame]: Dictionary with statistics for each point keyed by statistic label
    """
    stats_df_dict = {}
    stats = [s for s in extreme_statistics if s.stats in ["min", "max"]]
    if (
        len(set([s.stats for s in stats])) == len(stats) == 2
        and len(set([s.params.method_type for s in stats])) == 1
    ):
        extremes_df = extreme_values_analysis(
            params=stats[0].params,
            data_df=data_df,
            timestep_arr=timestep_arr,
        )
        stats_df_dict["min"] = extremes_df.iloc[0]
        stats_df_dict["max"] = extremes_df.iloc[1]
    else:
        for stat in stats:
            extremes_df = extreme_values_analysis(
                params=stat.params,
                data_df=data_df,
                timestep_arr=timestep_arr,
            )
            target_index = 0 if stat.stats == "min" else 1
            stats_df_dict[stat.stats] = extremes_df.iloc[target_index]

    return stats_df_dict


def calculate_mean_equivalent(
    statistics_to_apply: list[BasicStatisticModel | ParameterizedStatisticModel],
    stats_df_dict: dict[str, pd.Series],
) -> np.ndarray:
    """Calculates Mean Equivalent values, which are based on other stats such as min, max and mean.
    It uses the greater absolute value, check the docs for more details.

    Args:
        statistics_to_apply (list[BasicStatisticModel | ParameterizedStatisticModel]): List of statistical functions to apply
        stats_df_dict (dict[str, pd.Series]): Statistics series dictionary

    Returns:
        np.ndarray: Mean equivalent values array
    """
    comparison_df = pd.DataFrame()
    mean_eq_stat = [s for s in statistics_to_apply if s.stats == "mean_eq"][0]
    scale_factor = mean_eq_stat.params.scale_factor
    for stat_lbl in ["min", "max", "mean"]:
        comparison_df[stat_lbl] = stats_df_dict[stat_lbl].copy()
        comparison_df[stat_lbl] *= 1 if stat_lbl == "mean" else scale_factor

    max_abs_col_index = np.abs(comparison_df.values).argmax(axis=1)
    max_abs_values = comparison_df.values[np.arange(len(comparison_df)), max_abs_col_index]

    return max_abs_values


def calculate_statistics(
    historical_data: pd.DataFrame,
    statistics_to_apply: list[BasicStatisticModel | ParameterizedStatisticModel],
) -> pd.DataFrame:
    """Calculates statistics for force coefficient of a body data

    Args:
        historical_data (pd.DataFrame): Dataframe of the data coefficients historical series
        statistics_to_apply (list[BasicStatisticModel | ParameterizedStatisticModel]): List of statistical functions to apply

    Returns:
        pd.DataFrame: Statistics for the given coefficient
    """
    stats_df_dict: dict[str, pd.Series] = {}
    statistics_list = [s.stats for s in statistics_to_apply]
    data_df = historical_data.drop(columns=["time_normalized"])

    if "mean" in statistics_list:
        mean_df = data_df.mean()
        stats_df_dict["mean"] = mean_df
    if "rms" in statistics_list:
        rms_df = data_df.std()
        stats_df_dict["rms"] = rms_df
    if "skewness" in statistics_list:
        skewness_df = data_df.skew()
        stats_df_dict["skewness"] = skewness_df
    if "kurtosis" in statistics_list:
        kurtosis_df = data_df.kurt()
        stats_df_dict["kurtosis"] = kurtosis_df
    if "min" in statistics_list or "max" in statistics_list:
        stats = [s for s in statistics_to_apply if s.stats in ["min", "max"]]
        stats_df_dict = stats_df_dict | calculate_extreme_values(
            extreme_statistics=stats,
            timestep_arr=historical_data["time_normalized"].to_numpy(),
            data_df=data_df,
        )
    if "mean_eq" in statistics_list:
        stats_df_dict["mean_eq"] = calculate_mean_equivalent(
            statistics_to_apply=statistics_to_apply, stats_df_dict=stats_df_dict
        )

    return pd.DataFrame(stats_df_dict)


def combine_stats_data_with_mesh(
    mesh: LnasGeometry,
    region_idx_array: np.ndarray,
    data_stats: pd.DataFrame,
) -> pd.DataFrame:
    """Combine compiled statistical data with surface meshing by indexing regions

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
