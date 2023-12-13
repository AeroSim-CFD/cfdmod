from typing import Literal

import pandas as pd


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
