from typing import Literal

import pandas as pd

from cfdmod.use_cases.pressure.cp_config import Statistics


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

    press_data = press_data[
        (press_data["time_step"] >= timestep_range[0])
        & (press_data["time_step"] <= timestep_range[1])
    ]

    body_data = body_data[
        (body_data["time_step"] >= timestep_range[0])
        & (body_data["time_step"] <= timestep_range[1])
    ]

    return press_data, body_data


def transform_to_cp(
    press_data: pd.DataFrame,
    body_data: pd.DataFrame,
    reference_vel: float,
    ref_press_mode: Literal["instantaneous", "average"],
) -> pd.DataFrame:
    """Transform the body pressure data into Cp coefficient

    Args:
        press_data (pd.DataFrame): Historic series pressure DataFrame
        body_data (pd.DataFrame): Body's DataFrame
        reference_vel (float): Value of reference velocity for dynamic pressure
        ref_press_mode (Literal["instantaneous", "average"]): Sets how to account for reference pressure effects

    Returns:
        pd.DataFrame: Dataframe of pressure coefficient data for the body
    """

    average_static_pressure = press_data["rho"].to_numpy().mean()
    dynamic_pressure = 0.5 * average_static_pressure * reference_vel**2
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


def calculate_statistics(
    body_data: pd.DataFrame, statistics_to_apply: list[Statistics]
) -> pd.DataFrame:
    """Calculates statistics for pressure coefficient of a body data

    Args:
        body_data (pd.DataFrame): Dataframe of the body data pressure coefficients
        statistics_to_apply (Statistics): List of statistical functions to apply

    Returns:
        pd.DataFrame: Statistics for pressure coefficient
    """
    group_by_point_cp = body_data.groupby("point_idx")["cp"]

    statistics_data = pd.DataFrame({"point_idx": body_data["point_idx"].unique()})

    if "avg" in statistics_to_apply:
        statistics_data["cp_avg"] = group_by_point_cp.mean()
    if "min" in statistics_to_apply:
        statistics_data["cp_min"] = group_by_point_cp.min()
    if "max" in statistics_to_apply:
        statistics_data["cp_max"] = group_by_point_cp.max()
    if "std" in statistics_to_apply:
        statistics_data["cp_rms"] = group_by_point_cp.std()

    # Calculate skewness and kurtosis using apply
    if "skewness" in statistics_to_apply:
        skewness = group_by_point_cp.apply(lambda x: x.skew()).reset_index(name="skewness")
        statistics_data["cp_skewness"] = skewness["skewness"]
    if "kurtosis" in statistics_to_apply:
        kurtosis = group_by_point_cp.apply(lambda x: x.kurt()).reset_index(name="kurtosis")
        statistics_data["cp_kurtosis"] = kurtosis["kurtosis"]

    return statistics_data
