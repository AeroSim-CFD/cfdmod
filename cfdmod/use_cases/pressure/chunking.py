import math
import pathlib
from typing import Callable

import numpy as np
import pandas as pd
from lnas import LnasGeometry

from cfdmod.use_cases.pressure.extreme_values import ExtremeValuesParameters
from cfdmod.use_cases.pressure.statistics import Statistics
from cfdmod.use_cases.pressure.zoning.processing import calculate_statistics


def split_into_chunks(
    time_series_df: pd.DataFrame, number_of_chunks: int, output_path: pathlib.Path
):
    """Split time series data into chunks

    Args:
        time_series_df (pd.DataFrame): Time series dataframe
        number_of_chunks (int): Target number of chunks
        output_path (pathlib.Path): Output path

    Raises:
        ValueError: Raises error if dataframe is not a time series (time_step not in columns)
    """

    if "time_step" not in time_series_df.columns:
        raise ValueError("Time series dataframe must have a time_step column to be chunked")

    time_arr = time_series_df.time_step.unique()
    step = math.ceil(len(time_arr) / number_of_chunks)

    if len(time_arr) / number_of_chunks < 2:
        raise ValueError("There must be at least two steps in each chunk")

    for i in range(number_of_chunks):
        min_step, max_step = i * step, min((i + 1) * step - 1, len(time_arr) - 1)
        df: pd.DataFrame = time_series_df.loc[
            (time_series_df.time_step >= time_arr[min_step])
            & (time_series_df.time_step <= time_arr[max_step])
        ].copy()

        range_lbl = f"range_{int(time_arr[min_step])}_{int(time_arr[max_step])}"

        df.to_hdf(path_or_buf=output_path, key=range_lbl, mode="a", index=False, format="t")


def calculate_statistics_for_groups(
    grouped_data_path: pathlib.Path,
    statistics: list[Statistics],
    extreme_params: ExtremeValuesParameters | None,
) -> pd.DataFrame:
    """Calculates statistics for groups of points

    Args:
        grouped_data_path (pathlib.Path): Path of grouped data (HDF)
        statistics (list[Statistics]): List of statistics to apply
        extreme_params (ExtremeValuesParameters | None): Parameters for extreme values analysis

    Returns:
        pd.DataFrame: Statistics dataframe
    """
    stats_df = []

    with pd.HDFStore(grouped_data_path, mode="r") as groups_store:
        groups = groups_store.keys()
        point_groups = set(["group" + k.split("group")[1] for k in groups])

        for group_lbl in point_groups:
            keys_for_group = [k for k in groups if k.split("group")[1] in group_lbl]
            group_dfs = []
            for key in keys_for_group:
                df = groups_store.get(key)
                group_dfs.append(df)

            cp_data = pd.concat(group_dfs).sort_values(by=["time_step", "point_idx"])
            cp_stats = calculate_statistics(
                cp_data,
                statistics_to_apply=statistics,
                variables=["cp"],
                group_by_key="point_idx",
                extreme_params=extreme_params,
            )
            del cp_data
            stats_df.append(cp_stats)

    full_stats = pd.concat(stats_df).sort_values(by=["point_idx"])

    return full_stats


def divide_timeseries_in_groups(
    n_groups: int, timeseries_path: pathlib.Path, output_path: pathlib.Path
):
    """Divides timeseries into groups of points

    Args:
        n_groups (int): Number of point groups
        timeseries_path (pathlib.Path): Path to the timeseries
        output_path (pathlib.Path): Output path
    """
    with pd.HDFStore(timeseries_path, mode="r") as data_store:
        groups = data_store.keys()
        pt_groups = None

        for group_lbl in groups:
            coefficient_data = data_store.get(group_lbl)
            if pt_groups == None:
                points_arr = coefficient_data.point_idx.unique()
                n_per_group = len(points_arr) // n_groups
                pt_groups = np.split(points_arr, range(n_per_group, len(points_arr), n_per_group))

            for i, points_in_group in enumerate(pt_groups):
                group_data = coefficient_data.loc[
                    coefficient_data.point_idx.isin(points_in_group)
                ].copy()
                group_data.to_hdf(output_path, key=group_lbl + f"_group_{i:03}", mode="a")


def process_timestep_groups(
    data_path: pathlib.Path,
    geometry_df: pd.DataFrame,
    geometry: LnasGeometry,
    processing_function: Callable[[pd.DataFrame, pd.DataFrame, LnasGeometry], pd.DataFrame],
) -> pd.DataFrame:
    """Process the timestep groups with geometric properties

    Args:
        data_path (pathlib.Path): Path for pressure coefficient data
        geometry_df (pd.DataFrame): Geometric properties dataframe
        geometry (LnasGeometry): Geometry to be processed. Needed for evaluating representative area and volume
        processing_function (Callable[[pd.DataFrame, pd.DataFrame, LnasGeometry], pd.DataFrame]):
            Coefficient processing function

    Returns:
        pd.DataFrame: Transformed pressure coefficient time series
    """

    processed_samples: list[pd.DataFrame] = []
    with pd.HDFStore(data_path, mode="r") as df_store:
        store_groups = df_store.keys()

        for store_group in store_groups:
            sample = df_store.get(store_group)
            coefficient_data = processing_function(sample, geometry_df, geometry)
            processed_samples.append(coefficient_data)

    merged_samples = pd.concat(processed_samples)

    sort_columns = [col for col in ["time_step", "region_idx"] if col in merged_samples.columns]
    if "time_step" in sort_columns:
        merged_samples.sort_values(by=sort_columns, inplace=True)
    else:
        raise KeyError("Missing time_step column in data stored")

    return merged_samples
