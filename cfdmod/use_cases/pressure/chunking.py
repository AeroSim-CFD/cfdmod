import math
import pathlib
from typing import Callable, ClassVar

import numpy as np
import pandas as pd
from lnas import LnasGeometry

from cfdmod.use_cases.pressure.statistics import BasicStatisticModel, ParameterizedStatisticModel
from cfdmod.use_cases.pressure.zoning.processing import calculate_statistics
from cfdmod.utils import convert_matrix_into_dataframe, convert_dataframe_into_matrix


class HDFGroupInterface:
    # HDF keys follow the convention /step_{formatted initial_step}_group_{formatted group_idx}
    # Step information comes from simulation results
    TEMPORAL_PREFIX: ClassVar[str] = "/step"
    GROUP_PREFIX: ClassVar[str] = "group"

    @classmethod
    def time_key(cls, initial_time: float) -> str:
        return f"{cls.TEMPORAL_PREFIX}{int(initial_time):07}"

    @classmethod
    def get_point_group_key(cls, timestep_group_lbl: str, group_idx: int) -> str:
        return "_".join([timestep_group_lbl, cls.GROUP_PREFIX + f"{group_idx:04}"])

    @classmethod
    def get_available_point_groups(cls, hdf_keys: list[str]) -> set[str]:
        return set([cls.GROUP_PREFIX + k.split(cls.GROUP_PREFIX)[1] for k in hdf_keys])

    @classmethod
    def get_timestep_keys_for_group(cls, hdf_keys: list[str], group_key: str) -> list[str]:
        return [k for k in hdf_keys if k.split(cls.GROUP_PREFIX)[1] in group_key]

    @classmethod
    def filter_groups(
        cls, group_keys: list[str], timestep_range: tuple[float, float]
    ) -> list[str]:
        steps = [int(key.replace(cls.TEMPORAL_PREFIX, "")) for key in group_keys]
        lower_values = [x for x in steps if x <= timestep_range[0]]
        if len(lower_values) == 0:
            return [cls.time_key(step) for step in steps if step <= timestep_range[1]]
        else:
            initial_step = max(lower_values)
            return [
                cls.time_key(step) for step in steps if initial_step <= step <= timestep_range[1]
            ]


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
        initial_step, end_step = i * step, min((i + 1) * step - 1, len(time_arr) - 1)
        df: pd.DataFrame = time_series_df.loc[
            (time_series_df.time_step >= time_arr[initial_step])
            & (time_series_df.time_step <= time_arr[end_step])
        ].copy()

        range_lbl = HDFGroupInterface.time_key(initial_time=time_arr[initial_step])

        df.to_hdf(path_or_buf=output_path, key=range_lbl, mode="a", index=False, format="table")


def calculate_statistics_for_groups(
    grouped_data_path: pathlib.Path,
    statistics: list[BasicStatisticModel | ParameterizedStatisticModel],
) -> pd.DataFrame:
    """Calculates statistics for groups of points

    Args:
        grouped_data_path (pathlib.Path): Path of grouped data (HDF)
        statistics (list[BasicStatisticModel | ParameterizedStatisticModel]): List of statistics with parameters to apply

    Returns:
        pd.DataFrame: Statistics dataframe
    """
    stats_df = []

    with pd.HDFStore(grouped_data_path, mode="r") as groups_store:
        hdf_keys = groups_store.keys()
        point_groups = HDFGroupInterface.get_available_point_groups(hdf_keys)

        for group_lbl in point_groups:
            keys_for_group = HDFGroupInterface.get_timestep_keys_for_group(hdf_keys, group_lbl)
            group_dfs = []
            for key in keys_for_group:
                df = groups_store.get(key)
                group_dfs.append(df)
            cp_data = pd.concat(group_dfs).sort_values(by=["time_normalized"])
            cp_stats = calculate_statistics(
                cp_data,
                statistics_to_apply=statistics,
            )
            del cp_data
            stats_df.append(cp_stats)

    full_stats = pd.concat(stats_df).T
    full_stats.reset_index(inplace=True)
    full_stats.rename(columns={"index": "scalar"}, inplace=True)

    return full_stats[["scalar"] + sorted([col for col in full_stats.columns if col != "scalar"])]


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
            if pt_groups is None:
                points_arr = np.array(
                    [col for col in coefficient_data.columns if col != "time_normalized"],
                    dtype=np.int32,
                )
                n_per_group = len(points_arr) // n_groups
                pt_groups = np.split(points_arr, range(n_per_group, len(points_arr), n_per_group))

            for i, points_in_group in enumerate(pt_groups):
                group_data = coefficient_data[points_in_group.astype(str)].copy()
                group_data["time_normalized"] = coefficient_data["time_normalized"]
                group_key = HDFGroupInterface.get_point_group_key(group_lbl, i)
                group_data.to_hdf(output_path, key=group_key, mode="a", format="fixed")
                del group_data


def process_timestep_groups(
    data_path: pathlib.Path,
    geometry_df: pd.DataFrame,
    geometry: LnasGeometry,
    processing_function: Callable[[pd.DataFrame, pd.DataFrame, LnasGeometry], pd.DataFrame],
    data_label: str = "cp",
    time_column_label: str = "time_normalized",
) -> pd.DataFrame:
    """Process the timestep groups with geometric properties

    Args:
        data_path (pathlib.Path): Path for pressure coefficient data
        geometry_df (pd.DataFrame): Geometric properties dataframe
        geometry (LnasGeometry): Geometry to be processed. Needed for evaluating representative area and volume
        processing_function (Callable[[pd.DataFrame, pd.DataFrame, LnasGeometry], pd.DataFrame]):
            Coefficient processing function
        data_label (str): Label of the tabulated time series dataframe. Defaults to "cp".
        time_column_label (str): Label of time series time column. Defaults to "time_normalized".

    Returns:
        pd.DataFrame: Transformed pressure coefficient time series
    """

    processed_samples: list[pd.DataFrame] = []
    with pd.HDFStore(data_path, mode="r") as df_store:
        store_groups = df_store.keys()

        for store_group in store_groups:
            sample = df_store.get(store_group)
            
            list_of_functions_that_uses_matrix_format = ['transform_Cf']
            if "point_idx" in sample.columns and processing_function in list_of_functions_that_uses_matrix_format:
                sample = convert_dataframe_into_matrix(
                    sample, row_data_label=time_column_label, value_data_label=data_label
                )
            if "point_idx" not in sample.columns and processing_function not in list_of_functions_that_uses_matrix_format:
                # If point_idx is not in dataframe columns, then matrix form is assumed
                # and needs to be converted to older format
                sample = convert_matrix_into_dataframe(
                    sample, row_data_label=time_column_label, value_data_label=data_label
                )
                    
            coefficient_data = processing_function(sample, geometry_df, geometry)
            processed_samples.append(coefficient_data)

    merged_samples = pd.concat(processed_samples)
    merged_samples.rename(columns={col: str(col) for col in merged_samples.columns}, inplace=True)

    sort_columns = [
        col for col in [time_column_label, "region_idx"] if col in merged_samples.columns
    ]
    if time_column_label in sort_columns:
        merged_samples.sort_values(by=sort_columns, inplace=True)
    else:
        raise KeyError(f"Missing time {time_column_label} column in data stored")

    return merged_samples
