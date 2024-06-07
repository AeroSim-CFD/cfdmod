import pathlib
import warnings
from typing import Literal

import pandas as pd
from lnas import LnasGeometry

from cfdmod.api.vtk.write_vtk import create_polydata_for_cell_data, write_polydata
from cfdmod.logger import logger
from cfdmod.use_cases.pressure.chunking import (
    HDFGroupInterface,
    calculate_statistics_for_groups,
    divide_timeseries_in_groups,
)
from cfdmod.use_cases.pressure.cp_config import CpConfig
from cfdmod.use_cases.pressure.path_manager import CpPathManager
from cfdmod.utils import convert_dataframe_into_matrix, create_folders_for_file, save_yaml


def transform_to_cp(
    press_data: pd.DataFrame,
    body_data: pd.DataFrame,
    reference_vel: float,
    characteristic_length: float,
    ref_press_mode: Literal["instantaneous", "average"],
) -> pd.DataFrame:
    """Transform the body pressure data into Cp coefficient

    Args:
        press_data (pd.DataFrame): Historic series pressure DataFrame
        body_data (pd.DataFrame): Body's DataFrame
        reference_vel (float): Value of reference velocity for dynamic pressure
        characteristic_length (float): Characteristic length in simulation time scale
        ref_press_mode (Literal["instantaneous", "average"]): Sets how to account for reference pressure effects

    Returns:
        pd.DataFrame: Dataframe of pressure coefficient data for the body
    """
    static_pressure_array = press_data["0"].to_numpy()
    average_static_pressure = static_pressure_array.mean()
    dynamic_pressure = 0.5 * average_static_pressure * reference_vel**2
    cs_square = 1 / 3
    multiplier = cs_square / dynamic_pressure
    press = static_pressure_array if ref_press_mode == "instantaneous" else average_static_pressure

    columns_to_convert = [col for col in body_data.columns if col != "time_step"]
    data_to_convert = body_data[columns_to_convert].to_numpy()
    result = (data_to_convert.T - press) * multiplier
    df_cp = pd.DataFrame(result.T, columns=columns_to_convert)
    df_cp["time_normalized"] = body_data["time_step"].to_numpy() / (
        characteristic_length / reference_vel
    )

    return df_cp[
        ["time_normalized"]
        + [col for col in df_cp.columns if col not in ["time_step", "time_normalized"]]
    ]


def filter_data(data: pd.DataFrame, timestep_range: tuple[float, float]) -> pd.DataFrame:
    """Filter data in between timestep range

    Args:
        data (pd.DataFrame): Dataframe to be filtered
        timestep_range (tuple[float, float]): Range of timestep to filter data

    Returns:
        pd.DataFrame: Data filtered
    """

    filtered_data = data[
        (data["time_step"] >= timestep_range[0]) & (data["time_step"] <= timestep_range[1])
    ].copy()

    return filtered_data


def process_raw_groups(
    static_pressure_path: pathlib.Path,
    body_pressure_path: pathlib.Path,
    output_path: pathlib.Path,
    cp_config: CpConfig,
):
    """Saves transformed data (pressure coefficient) into time series and a grouped data.

    Args:
        static_pressure_path (pathlib.Path): Path of the static pressure time series
        body_pressure_path (pathlib.Path): Path of the body pressure time series
        output_path (pathlib.Path): Output path of the timeseries
        cp_config (CpConfig): Pressure coefficient configuration

    Raises:
        Exception: If the keys for body and static pressure data do not match
    """

    def check_numeric(value) -> bool:
        try:
            float(value)
            return True
        except:
            return False

    with pd.HDFStore(body_pressure_path, mode="r") as body_store:
        with pd.HDFStore(static_pressure_path, mode="r") as static_store:
            static_groups = static_store.keys()
            body_groups = body_store.keys()

            if static_groups != body_groups:
                raise Exception(f"Keys for body and static pressure don't match!")

            more_than_one_group = len(body_groups) > 1

            keys_to_include: list[str] = []

            if more_than_one_group:
                keys_to_include = HDFGroupInterface.filter_groups(
                    body_groups, cp_config.timestep_range
                )

            average_value = None

            if cp_config.reference_pressure == "average":
                static_dfs = []
                for store_group in static_groups:
                    static_dfs.append(static_store.get(store_group))
                merged_df = pd.concat(static_dfs)
                merged_df.rename(
                    columns={
                        col: str(int(float(col)))
                        for col in merged_df.columns
                        if str(col).isnumeric()
                    },
                    inplace=True,
                )
                # Old versions index the column with rho and new versions use point index (0)
                # to label the column. Hence the condition below
                average_value = (
                    merged_df["rho"].mean()
                    if "rho" in merged_df.columns
                    else merged_df["0"].mean()
                )

            for store_group in body_groups:
                if more_than_one_group and store_group not in keys_to_include:
                    continue

                static_df = static_store.get(store_group)
                static_df = filter_data(static_df, timestep_range=cp_config.timestep_range)

                body_df = body_store.get(store_group)
                body_df = filter_data(body_df, timestep_range=cp_config.timestep_range)

                # FIX CONVERSION ERROR
                static_df.rename(
                    columns={
                        col: str(int(float(col)))
                        for col in static_df.columns
                        if check_numeric(col)
                    },
                    inplace=True,
                )
                body_df.rename(
                    columns={
                        col: str(int(float(col))) for col in body_df.columns if check_numeric(col)
                    },
                    inplace=True,
                )

                # This logic should be removed in later versions
                if "point_idx" in body_df.columns:
                    # Data is in older format, must convert to matrix
                    body_df = convert_dataframe_into_matrix(body_df)
                if "point_idx" in static_df.columns:
                    # Data is in older format, must convert to matrix
                    static_df = convert_dataframe_into_matrix(static_df)

                if average_value != None:
                    static_df["0"] = average_value

                if any(static_df.time_step.unique() != body_df.time_step.unique()):
                    raise Exception(f"Timesteps for key {store_group} do not match!")

                coefficient_data = transform_to_cp(
                    press_data=static_df,
                    body_data=body_df,
                    reference_vel=cp_config.simul_U_H,
                    characteristic_length=cp_config.simul_characteristic_length,
                    ref_press_mode=cp_config.reference_pressure,
                )
                coefficient_data.rename(
                    columns={col: str(col) for col in coefficient_data.columns}, inplace=True
                )
                coefficient_data.to_hdf(output_path, key=store_group, mode="a", format="fixed")


def process_cp(
    pressure_data_path: pathlib.Path,
    body_data_path: pathlib.Path,
    cfg_label: str,
    cfg: CpConfig,
    mesh: LnasGeometry,
    path_manager: CpPathManager,
):
    """Executes the pressure coefficient processing routine

    Args:
        pressure_data_path (pathlib.Path): Path for static reference pressure time series
        body_data_path (pathlib.Path): Path for body pressure time series
        cfg_label (str): Label of the configuration
        cfg (CpConfig): Pressure coefficient configuration
        mesh (LnasGeometry): Geometry of the body
        path_manager (CpPathManager): Object to handle paths
    """
    timeseries_path = path_manager.get_timeseries_path(cfg_lbl=cfg_label)
    create_folders_for_file(timeseries_path)

    if timeseries_path.exists():
        warnings.warn(
            f"Path for time series already exists {timeseries_path}. Deleted old file",
            RuntimeWarning,
        )
        timeseries_path.unlink()

    logger.info("Transforming into pressure coefficient")
    process_raw_groups(
        static_pressure_path=pressure_data_path,
        body_pressure_path=body_data_path,
        output_path=timeseries_path,
        cp_config=cfg,
    )

    grouped_data_path = path_manager.get_grouped_timeseries_path(cfg_lbl=cfg_label)

    if grouped_data_path.exists():
        warnings.warn(
            f"Path for grouped time series already exists {grouped_data_path}. Deleted old file",
            RuntimeWarning,
        )
        grouped_data_path.unlink()

    logger.info("Dividing into point groups")
    divide_timeseries_in_groups(
        n_groups=cfg.number_of_chunks,
        timeseries_path=timeseries_path,
        output_path=grouped_data_path,
    )

    logger.info("Calculating statistics")
    cp_stats = calculate_statistics_for_groups(
        grouped_data_path=grouped_data_path,
        statistics=cfg.statistics,
    )
    stats_path = path_manager.get_stats_path(cfg_lbl=cfg_label)
    cp_stats.to_hdf(path_or_buf=stats_path, key="stats", mode="w", index=False, format="fixed")

    logger.info("Exporting files")
    save_yaml(cfg.model_dump(), path_manager.get_config_path(cfg_lbl=cfg_label))
    vtp_path = path_manager.get_vtp_path(cfg_lbl=cfg_label)
    polydata = create_polydata_for_cell_data(data=cp_stats, mesh=mesh)
    write_polydata(vtp_path, polydata)
