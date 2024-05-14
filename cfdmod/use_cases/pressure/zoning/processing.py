from typing import Literal, Optional

import numpy as np
import pandas as pd
from lnas import LnasGeometry

from cfdmod.use_cases.pressure.extreme_values import (  # ExtremeValuesParameters,
    gumbel_extreme_values,
    moving_average_extreme_values,
)
from cfdmod.use_cases.pressure.statistics import Statistics

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


def perform_extreme_value_analysis(
    historical_data: pd.DataFrame,
    statistics_to_apply: list[Statistics],
    var_name: ShapeVariables | ForceVariables | MomentVariables | PressureVariables,
    # extreme_params: ExtremeValuesParameters,
    statistics_data: pd.DataFrame,
    group_by_key: str,
):
    """Perform extreme values analysis to historical data and add to statistics data

    Args:
        historical_data (pd.DataFrame): Time series data
        statistics_to_apply (list[Statistics]): List of statistics to apply
        var_name (ShapeVariables | ForceVariables | MomentVariables | PressureVariables): Current
            variable being processed
        extreme_params (ExtremeValuesParameters): Parameters for extreme value analysis
        statistics_data (pd.DataFrame): Compiled statistics data
        group_by_key (str): Key to identify a parameter for grouping
    """

    def _get_mean_peak_value(row: pd.Series, variable: str) -> float:
        factor = extreme_params.time_scale_correction_factor
        possible_values = [
            row[f"{variable}_mean"],
            factor * row[f"{variable}_xtr_max"],
            factor * row[f"{variable}_xtr_min"],
        ]
        max_abs_value = max(possible_values, key=abs)
        max_abs_index = possible_values.index(max_abs_value)

        if max_abs_value == 0:
            return 0
        else:
            return max_abs_value * possible_values[max_abs_index] / max_abs_value

    group_by_point = historical_data.groupby(group_by_key)
    timestep = historical_data.time_step.unique()

    if extreme_params.extreme_model == "Moving average":
        xtr_stats = (
            group_by_point[var_name]
            .apply(lambda x: moving_average_extreme_values(params=extreme_params, hist_series=x))
            .reset_index(name="xtr_val")
        )
    elif extreme_params.extreme_model == "Gumbel":
        xtr_stats = (
            group_by_point[var_name]
            .apply(
                lambda x: gumbel_extreme_values(
                    params=extreme_params, timestep_arr=timestep, hist_series=x
                )
            )
            .reset_index(name="xtr_val")
        )
    else:
        raise Exception(f"Unknown extreme values model {extreme_params.extreme_model}")

    statistics_data[[f"{var_name}_xtr_min", f"{var_name}_xtr_max"]] = xtr_stats["xtr_val"].apply(
        lambda x: pd.Series(x)
    )
    if "mean_eq" in statistics_to_apply:
        mean_eq = statistics_data.apply(
            lambda x: _get_mean_peak_value(x, var_name), axis=1
        ).reset_index(name="mean_eq")
        statistics_data[f"{var_name}_mean_eq"] = mean_eq["mean_eq"]
        if "mean" not in statistics_to_apply:
            statistics_data = statistics_data.drop(f"{var_name}_mean", axis=1)
    if "xtr_min" not in statistics_to_apply:
        statistics_data = statistics_data.drop(f"{var_name}_xtr_min", axis=1)
    if "xtr_max" not in statistics_to_apply:
        statistics_data = statistics_data.drop(f"{var_name}_xtr_max", axis=1)


def calculate_statistics(
    historical_data: pd.DataFrame,
    statistics_to_apply: list[Statistics],
    variables: (
        list[ShapeVariables]
        | list[ForceVariables]
        | list[MomentVariables]
        | list[PressureVariables]
    ),
    group_by_key: str,
    # extreme_params: Optional[ExtremeValuesParameters] = None,
) -> pd.DataFrame:
    """Calculates statistics for force coefficient of a body data

    Args:
        historical_data (pd.DataFrame): Dataframe of the data coefficients historical series
        statistics_to_apply (list[Statistics]): List of statistical functions to apply
        variables (list[str]): List of variables to apply statistical analysis
        group_by_key (str): Key to identify a parameter for grouping
        extreme_params (Optional[ExtremeValuesParameters]): Parameters for extreme values analysis. Defaults to None.

    Returns:
        pd.DataFrame: Statistics for the given coefficient
    """
    group_by_point = historical_data.groupby(group_by_key)
    statistics_data = pd.DataFrame({group_by_key: historical_data[group_by_key].unique()})

    for var_name in variables:
        if "mean" in statistics_to_apply or "mean_eq" in statistics_to_apply:
            average = group_by_point[var_name].apply(lambda x: x.mean()).reset_index(name="mean")
            statistics_data[f"{var_name}_mean"] = average["mean"]
        if "min" in statistics_to_apply:
            minimum = group_by_point[var_name].apply(lambda x: x.min()).reset_index(name="min")
            statistics_data[f"{var_name}_min"] = minimum["min"]
        if "max" in statistics_to_apply:
            maximum = group_by_point[var_name].apply(lambda x: x.max()).reset_index(name="max")
            statistics_data[f"{var_name}_max"] = maximum["max"]
        if "std" in statistics_to_apply:
            std = group_by_point[var_name].apply(lambda x: x.std()).reset_index(name="std")
            statistics_data[f"{var_name}_std"] = std["std"]

        # Calculate skewness and kurtosis using apply
        if "skewness" in statistics_to_apply:
            skewness = (
                group_by_point[var_name].apply(lambda x: x.skew()).reset_index(name="skewness")
            )
            statistics_data[f"{var_name}_skewness"] = skewness["skewness"]
        if "kurtosis" in statistics_to_apply:
            kurtosis = (
                group_by_point[var_name].apply(lambda x: x.kurt()).reset_index(name="kurtosis")
            )
            statistics_data[f"{var_name}_kurtosis"] = kurtosis["kurtosis"]

        # Extreme values analysis
        if (
            any([v in statistics_to_apply for v in ["xtr_min", "xtr_max"]])
            or "mean_eq" in statistics_to_apply
        ):
            if extreme_params is None:
                raise ValueError("Missing extreme values parameters!")
            perform_extreme_value_analysis(
                historical_data=historical_data,
                statistics_to_apply=statistics_to_apply,
                var_name=var_name,
                extreme_params=extreme_params,
                statistics_data=statistics_data,
                group_by_key=group_by_key,
            )

    return statistics_data


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
