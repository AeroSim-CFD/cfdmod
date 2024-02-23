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
from cfdmod.use_cases.pressure.extreme_values import ExtremeValuesParameters
from cfdmod.use_cases.pressure.path_manager import CpPathManager
from cfdmod.utils import create_folders_for_file


def transform_to_cp(
    press_data: pd.DataFrame,
    body_data: pd.DataFrame,
    reference_vel: float,
    ref_press_mode: Literal["instantaneous", "average"],
    correction_factor: float = 1,
) -> pd.DataFrame:
    """Transform the body pressure data into Cp coefficient

    Args:
        press_data (pd.DataFrame): Historic series pressure DataFrame
        body_data (pd.DataFrame): Body's DataFrame
        reference_vel (float): Value of reference velocity for dynamic pressure
        ref_press_mode (Literal["instantaneous", "average"]): Sets how to account for reference pressure effects
        correction_factor (float, optional): Reference Velocity correction factor. Defaults to 1.

    Returns:
        pd.DataFrame: Dataframe of pressure coefficient data for the body
    """
    average_static_pressure = press_data["rho"].to_numpy().mean()
    dynamic_pressure = 0.5 * average_static_pressure * (reference_vel * correction_factor) ** 2
    cs_square = 1 / 3
    multiplier = cs_square / dynamic_pressure

    df_pressure = press_data.set_index("time_step")
    df_body = body_data.set_index("time_step")

    if ref_press_mode == "instantaneous":
        df_body["cp"] = multiplier * (df_body["rho"] - df_body.index.map(df_pressure["rho"]))
    elif ref_press_mode == "average":
        df_body["cp"] = multiplier * (df_body["rho"] - average_static_pressure)

    df_body.reset_index(inplace=True)
    df_body.drop(columns=["rho"], inplace=True)

    return df_body


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
            for store_group in body_groups:
                if more_than_one_group:
                    if store_group not in keys_to_include:
                        continue

                static_df = static_store.get(store_group)
                static_df = filter_data(static_df, timestep_range=cp_config.timestep_range)
                body_df = body_store.get(store_group)
                body_df = filter_data(body_df, timestep_range=cp_config.timestep_range)

                if (static_df.time_step.unique() != body_df.time_step.unique()).all():
                    raise Exception(f"Timesteps for key {store_group} do not match!")

                coefficient_data = transform_to_cp(
                    press_data=static_df,
                    body_data=body_df,
                    reference_vel=cp_config.U_H,
                    ref_press_mode=cp_config.reference_pressure,
                    correction_factor=cp_config.U_H_correction_factor,
                )
                coefficient_data.to_hdf(output_path, key=store_group, mode="a")


def process_cp(
    pressure_data_path: pathlib.Path,
    body_data_path: pathlib.Path,
    cfg_label: str,
    cfg: CpConfig,
    mesh: LnasGeometry,
    path_manager: CpPathManager,
    extreme_params: ExtremeValuesParameters | None,
):
    """Executes the pressure coefficient processing routine

    Args:
        pressure_data_path (pathlib.Path): Path for static reference pressure time series
        body_data_path (pathlib.Path): Path for body pressure time series
        cfg_label (str): Label of the configuration
        cfg (CpConfig): Pressure coefficient configuration
        mesh (LnasGeometry): Geometry of the body
        path_manager (CpPathManager): Object to handle paths
        extreme_params (ExtremeValuesParameters | None): Optional parameters for extreme values analysis
    """
    cfg_hash = cfg.sha256()
    timeseries_path = path_manager.get_cp_t_path(cfg_lbl=cfg_label, cfg_hash=cfg_hash)
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

    grouped_data_path = path_manager.get_grouped_cp_path(cfg_lbl=cfg_label, cfg_hash=cfg_hash)

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
        extreme_params=extreme_params,
    )

    stats_path = path_manager.get_cp_stats_path(cfg_lbl=cfg_label, cfg_hash=cfg_hash)
    cp_stats.to_hdf(path_or_buf=stats_path, key="cp_stats", mode="w", index=False)

    vtp_path = path_manager.get_vtp_path(cfg_lbl=cfg_label, cfg_hash=cfg_hash)
    polydata = create_polydata_for_cell_data(data=cp_stats, mesh=mesh)
    write_polydata(vtp_path, polydata)
