import math
import pathlib
from typing import Callable

import pandas as pd


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

    if output_path.exists():
        output_path.unlink()  # Overwrite existing file

    for i in range(number_of_chunks):
        if (i + 1) * step - 1 > len(time_arr) - 1:
            time_range = [time_arr[i * step], time_arr[-1]]
        else:
            time_range = [time_arr[i * step], time_arr[(i + 1) * step - 1]]

        df: pd.DataFrame = time_series_df.loc[
            (time_series_df.time_step >= time_range[0])
            & (time_series_df.time_step <= time_range[1])
        ].copy()

        range_lbl = f"range_{int(time_range[0])}_{int(time_range[1])}"

        df.to_hdf(output_path, key=range_lbl, mode="a", index=False, format="t")


def process_timestep_groups(
    data_path: pathlib.Path,
    geometry_df: pd.DataFrame,
    processing_function: Callable[[pd.DataFrame], pd.DataFrame],
) -> pd.DataFrame:
    """Process the timestep groups with geometric properties

    Args:
        data_path (pathlib.Path): Path for pressure coefficient data
        geometry_df (pd.DataFrame): Geometric properties dataframe
        processing_function (Callable[[pd.DataFrame], pd.DataFrame]): Coefficient processing function

    Returns:
        pd.DataFrame: Transformed pressure coefficient time series
    """
    processed_samples: list[pd.DataFrame] = []
    with pd.HDFStore(data_path, mode="r") as df_store:
        store_groups = df_store.keys()

        for store_group in store_groups:
            sample = df_store.get(store_group)
            coefficient_data = processing_function(sample, geometry_df)
            processed_samples.append(coefficient_data)

    merged_samples = pd.concat(processed_samples)

    return merged_samples
