from dataclasses import dataclass
from typing import Literal

import pandas as pd
from lnas import LnasGeometry
from vtk import vtkPolyData

from cfdmod.api.vtk.write_vtk import create_polydata_for_cell_data, write_polydata
from cfdmod.use_cases.pressure.chunking import split_into_chunks
from cfdmod.use_cases.pressure.cp_config import CpConfig
from cfdmod.use_cases.pressure.extreme_values import ExtremeValuesParameters
from cfdmod.use_cases.pressure.path_manager import CpPathManager
from cfdmod.use_cases.pressure.zoning.processing import calculate_statistics
from cfdmod.utils import create_folders_for_file


@dataclass
class CpOutputs:
    cp_data: pd.DataFrame
    cp_stats: pd.DataFrame
    polydata: vtkPolyData

    def save_outputs(self, cfg: CpConfig, cfg_label: str, path_manager: CpPathManager):
        # Output 1: cp(t)
        timeseries_path = path_manager.get_cp_t_path(cfg_label=cfg_label)
        create_folders_for_file(timeseries_path)

        if timeseries_path.exists():
            timeseries_path.unlink()  # Overwrite existing file

        split_into_chunks(
            time_series_df=self.cp_data,
            number_of_chunks=cfg.number_of_chunks,
            output_path=timeseries_path,
        )

        # Output 2: cp stats
        stats_path = path_manager.get_cp_stats_path(cfg_label=cfg_label)
        create_folders_for_file(stats_path)
        self.cp_stats.to_hdf(stats_path, key="cp_stats", mode="w", index=False)

        # Output 3: VTK cp_stats
        vtp_path = path_manager.get_vtp_path(cfg_label=cfg_label)
        create_folders_for_file(vtp_path)
        write_polydata(vtp_path, self.polydata)


def filter_pressure_data(
    press_data: pd.DataFrame,
    body_data: pd.DataFrame,
    timestep_range: tuple[float, float],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Filter slice data

    Args:
        press_data (pd.DataFrame): Pressure dataframe
        body_data (pd.DataFrame): Path for body pressure data
        timestep_range (tuple[float, float]): Range of timestep to slice data

    Returns:
        tuple[pd.DataFrame, pd.DataFrame]: Tuple with static pressure data and body pressure data sliced
    """

    filtered_press_data = press_data[
        (press_data["time_step"] >= timestep_range[0])
        & (press_data["time_step"] <= timestep_range[1])
    ].copy()

    filtered_body_data = body_data[
        (body_data["time_step"] >= timestep_range[0])
        & (body_data["time_step"] <= timestep_range[1])
    ].copy()

    return filtered_press_data, filtered_body_data


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


def process_cp(
    pressure_data: pd.DataFrame,
    body_data: pd.DataFrame,
    cfg: CpConfig,
    mesh: LnasGeometry,
    extreme_params: ExtremeValuesParameters | None,
) -> CpOutputs:
    """Executes the pressure coefficient processing routine

    Args:
        pressure_data (pd.DataFrame): Static reference pressure time series
        body_data (pd.DataFrame): Body pressure time series
        cfg (CpConfig): Pressure coefficient configuration
        extreme_params (ExtremeValuesParameters | None): Optional parameters for extreme values analysis
        mesh (LnasGeometry): Geometry of the body

    Returns:
        CpOutputs: Compiled outputs for pressure coefficient use case
    """
    press_data, body_data = filter_pressure_data(pressure_data, body_data, cfg.timestep_range)

    cp_data = transform_to_cp(
        press_data,
        body_data,
        reference_vel=cfg.U_H,
        ref_press_mode=cfg.reference_pressure,
        correction_factor=cfg.U_H_correction_factor,
    )

    cp_stats = calculate_statistics(
        cp_data,
        statistics_to_apply=cfg.statistics,
        variables=["cp"],
        group_by_key="point_idx",
        extreme_params=extreme_params,
    )

    polydata = create_polydata_for_cell_data(data=cp_stats, mesh=mesh)

    cp_output = CpOutputs(cp_data=cp_data, cp_stats=cp_stats, polydata=polydata)

    return cp_output
