import math
import pathlib

import numpy as np
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
        # Overwrite existing file
        output_path.unlink()

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


def join_chunks_for_points(time_series_path: pathlib.Path, point_idxs: np.ndarray) -> pd.DataFrame:
    """Join chunks of timesteps filtering based on a array of indices

    Args:
        time_series_path (pathlib.Path): Path of the chunked time series
        point_idxs (np.ndarray): Array of indices

    Returns:
        pd.DataFrame: Joined time series
    """
    # target_idxs = np.array2string(point_idxs, separator=",")
    # print(target_idxs)
    dfs = []

    with pd.HDFStore(time_series_path, mode="r") as chunk_store:
        for data_lbl in chunk_store.keys():
            # df_restored = chunk_store.select(key=data_lbl, where=f"point_idx={target_idxs}")
            # dfs.append(df_restored)
            df_restored = chunk_store.get(key=data_lbl)
            filtered_df = df_restored[df_restored["point_idx"].isin(point_idxs)].copy()
            dfs.append(filtered_df)

    joined_df = pd.concat(dfs).sort_values(by=["time_step", "point_idx"])

    return joined_df
