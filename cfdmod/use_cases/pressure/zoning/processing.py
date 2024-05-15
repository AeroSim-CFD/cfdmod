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
    ExtremeMethods,
    ParameterizedStatisticModel,
    StatisticsParamsModel,
)

ForceVariables = Literal["Cfx", "Cfy", "Cfz"]
MomentVariables = Literal["Cmx", "Cmy", "Cmz"]
ShapeVariables = Literal["Ce"]
PressureVariables = Literal["cp"]


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


# def perform_extreme_value_analysis(
#     historical_data: pd.DataFrame,
#     statistics_to_apply: list[Statistics],
#     var_name: ShapeVariables | ForceVariables | MomentVariables | PressureVariables,
#     # extreme_params: ExtremeValuesParameters,
#     statistics_data: pd.DataFrame,
#     group_by_key: str,
# ):
#     """Perform extreme values analysis to historical data and add to statistics data

#     Args:
#         historical_data (pd.DataFrame): Time series data
#         statistics_to_apply (list[Statistics]): List of statistics to apply
#         var_name (ShapeVariables | ForceVariables | MomentVariables | PressureVariables): Current
#             variable being processed
#         extreme_params (ExtremeValuesParameters): Parameters for extreme value analysis
#         statistics_data (pd.DataFrame): Compiled statistics data
#         group_by_key (str): Key to identify a parameter for grouping
#     """

#     def _get_mean_peak_value(row: pd.Series, variable: str) -> float:
#         factor = extreme_params.time_scale_correction_factor
#         possible_values = [
#             row[f"{variable}_mean"],
#             factor * row[f"{variable}_xtr_max"],
#             factor * row[f"{variable}_xtr_min"],
#         ]
#         max_abs_value = max(possible_values, key=abs)
#         max_abs_index = possible_values.index(max_abs_value)

#         if max_abs_value == 0:
#             return 0
#         else:
#             return max_abs_value * possible_values[max_abs_index] / max_abs_value

#     group_by_point = historical_data.groupby(group_by_key)
#     timestep = historical_data.time_step.unique()

#     if extreme_params.extreme_model == "Moving average":
#         xtr_stats = (
#             group_by_point[var_name]
#             .apply(lambda x: moving_average_extreme_values(params=extreme_params, hist_series=x))
#             .reset_index(name="xtr_val")
#         )
#     elif extreme_params.extreme_model == "Gumbel":
#         xtr_stats = (
#             group_by_point[var_name]
#             .apply(
#                 lambda x: gumbel_extreme_values(
#                     params=extreme_params, timestep_arr=timestep, hist_series=x
#                 )
#             )
#             .reset_index(name="xtr_val")
#         )
#     else:
#         raise Exception(f"Unknown extreme values model {extreme_params.extreme_model}")

#     statistics_data[[f"{var_name}_xtr_min", f"{var_name}_xtr_max"]] = xtr_stats["xtr_val"].apply(
#         lambda x: pd.Series(x)
#     )
#     if "mean_eq" in statistics_to_apply:
#         mean_eq = statistics_data.apply(
#             lambda x: _get_mean_peak_value(x, var_name), axis=1
#         ).reset_index(name="mean_eq")
#         statistics_data[f"{var_name}_mean_eq"] = mean_eq["mean_eq"]
#         if "mean" not in statistics_to_apply:
#             statistics_data = statistics_data.drop(f"{var_name}_mean", axis=1)
#     if "xtr_min" not in statistics_to_apply:
#         statistics_data = statistics_data.drop(f"{var_name}_xtr_min", axis=1)
#     if "xtr_max" not in statistics_to_apply:
#         statistics_data = statistics_data.drop(f"{var_name}_xtr_max", axis=1)


def extreme_values_analysis(
    params: StatisticsParamsModel,
    data_df: pd.DataFrame,
    timestep_arr: np.ndarray = None,
    time_scale_factor: float = 1,
):
    stat_df = pd.DataFrame()
    if params.method_type == "Absolute":
        stat_df = data_df.apply(lambda x: (x.min(), x.max()))
    elif params.method_type == "Gumbel":
        stat_df = data_df.apply(
            lambda x: gumbel_extreme_values(
                params=params,
                time_scale_factor=time_scale_factor,
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
                time_scale_factor=time_scale_factor,
                hist_series=x,
            )
        )
    return stat_df


def calculate_statistics(
    historical_data: pd.DataFrame,
    statistics_to_apply: list[BasicStatisticModel | ParameterizedStatisticModel],
    time_scale_factor: float,
) -> pd.DataFrame:
    """Calculates statistics for force coefficient of a body data

    Args:
        historical_data (pd.DataFrame): Dataframe of the data coefficients historical series
        statistics_to_apply (list[BasicStatisticModel | ParameterizedStatisticModel]): List of statistical functions to apply
        time_scale_factor (float): Factor for converting time scales from CST values

    Returns:
        pd.DataFrame: Statistics for the given coefficient
    """
    stats_df_dict: dict[str, pd.DataFrame] = {}
    statistics_list = [s.stats for s in statistics_to_apply]
    data_df = historical_data.drop(columns=["time_step"])

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
        if (
            len(set([s.stats for s in stats])) == len(stats) == 2
            and len(set([s.params.method_type for s in stats])) == 1
        ):
            method_type = stats[0].params.method_type
            timestep_arr, scale_factor = None, None
            if method_type in ["Gumbel", "Moving Average"]:
                scale_factor = time_scale_factor
                if method_type in ["Gumbel"]:
                    timestep_arr = historical_data.time_step.to_numpy()
            extremes_df = extreme_values_analysis(
                params=stats[0].params,
                data_df=data_df,
                timestep_arr=timestep_arr,
                time_scale_factor=scale_factor,
            )
            stats_df_dict["min"] = extremes_df.iloc[0]
            stats_df_dict["max"] = extremes_df.iloc[1]
        else:
            for stat in stats:
                timestep_arr, scale_factor = None, None
                if stat.params.method_type in ["Gumbel", "Moving Average"]:
                    scale_factor = time_scale_factor
                    if stat.params.method_type in ["Gumbel"]:
                        timestep_arr = historical_data.time_step.to_numpy()
                extremes_df = extreme_values_analysis(
                    params=stat.params,
                    data_df=data_df,
                    timestep_arr=timestep_arr,
                    time_scale_factor=scale_factor,
                )
                target_index = 0 if stat.stats == "min" else 1
                stats_df_dict[stat.stats] = extremes_df.iloc[target_index]

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
