from typing import Literal

import numpy as np
import pandas as pd
import scipy
from scipy.ndimage import gaussian_filter

from cfdmod.analysis.inflow.profile import InflowData, NormalizationParameters

VelocityComponents = Literal["ux", "uy", "uz"]


def spectral_density_function(
    velocity_signal: np.ndarray,
    timestamps: np.ndarray,
    reference_velocity: float,
    characteristic_length: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Perform a FFT over a velocity signal

    Args:
        velocity_signal (np.ndarray): Array of instantaneous velocity signal
        timestamps (np.ndarray): Array of timestamps of the signal
        reference_velocity (float): Value for reference velocity. For normalization
        characteristic_length (float): Value for Characteristic length. For normalization

    Returns:
        tuple[np.ndarray, np.ndarray]: Tuple with spectral density values array and normalized frequency values array
    """

    def filter_avg_data(data: np.ndarray) -> np.ndarray:
        filtered_data = gaussian_filter(data, sigma=3)  # Sigma smooths the curve
        return filtered_data

    delta_t = timestamps[1] - timestamps[0]
    signal_frequency = 1 / delta_t

    (xf, yf) = scipy.signal.periodogram(velocity_signal, signal_frequency, scaling="density")
    st = np.std(velocity_signal)
    yf = xf * yf / st**2
    xf = xf * characteristic_length / reference_velocity  # Stroulhall number N = f * L / U

    # Get the filter coefficients so we can check its frequency response.
    yf = filter_avg_data(yf)
    return xf[2:], yf[2:]


def calculate_mean_velocity(
    inflow_data: InflowData, for_components: list[VelocityComponents]
) -> pd.DataFrame:
    """Calculates the turbulence intensity for each component given
    The inflow data dataframe must contain the columns selected by for_components (ux, uy, uz)

    Args:
        inflow_data (InflowData): Inflow data structure containing points and hist series
        for_components (list[VelocityComponents]): List of components to calculate mean velocity from

    Returns:
        pd.DataFrame: Mean Velocity dataframe with columns for each for_components suffixed by *_mean
    """
    if not all(
        [component in inflow_data.data.columns for component in for_components + ["point_idx"]]
    ):
        raise ValueError("Components must be inside inflow profile data columns")

    group_by_point_idx = inflow_data.data.groupby("point_idx")
    velocity_data = group_by_point_idx.agg(
        {component: "mean" for component in for_components}
    ).reset_index()
    # Rename columns
    velocity_data.columns = [
        col + "_mean" if col in for_components else col for col in velocity_data.columns
    ]
    return velocity_data


def calculate_turbulence_intensity(
    inflow_data: InflowData, for_components: list[VelocityComponents]
) -> pd.DataFrame:
    """Calculates the turbulence intensity for each component given
    The inflow data dataframe must contain the columns selected by for_components (ux, uy, uz)

    Args:
        inflow_data (InflowData): Inflow data structure containing points and hist series
        for_components (list[VelocityComponents]): List of components to calculate turbulence intensity from

    Returns:
        pd.DataFrame: Turbulence intensity dataframe with columns for each for_components preffixed by I_*
    """
    if not all(
        [component in inflow_data.data.columns for component in for_components + ["point_idx"]]
    ):
        raise ValueError("Components must be inside inflow profile data columns")

    group_by_point_idx = inflow_data.data.groupby("point_idx")
    turbulence_data = group_by_point_idx.agg(
        {component: ["mean", "std"] for component in for_components}
    ).reset_index()
    # Rename columns
    turbulence_data.columns = [
        "_".join(col) if col[1] != "" else col[0] for col in turbulence_data.columns
    ]
    for component in for_components:
        turbulence_data[f"I_{component}"] = (
            turbulence_data[f"{component}_std"] / turbulence_data[f"{component}_mean"]
        )

    return turbulence_data[
        ["point_idx"] + [f"I_{component}" for component in for_components]
    ].copy()


def calculate_spectral_density(
    inflow_data: InflowData,
    target_index: int,
    for_components: list[VelocityComponents],
    normalization_params: NormalizationParameters,
) -> pd.DataFrame:
    """Calculates the spectral density for a given target point index
    The inflow data dataframe must contain the columns selected by for_components (ux, uy, uz)

    Args:
        inflow_data (InflowData): Inflow data structure containing points and hist series
        target_index (int): Index of the target point
        for_components (list[VelocityComponents]): List of components to calculate turbulence intensity from
        normalization_params (NormalizationParameters): Parameters for spectral density normalization

    Returns:
        pd.DataFrame: Spectral density data with columns S (*) and f (*) for each for_components
    """
    spectral_data = pd.DataFrame()
    for component in for_components:
        point_data = inflow_data.data.loc[inflow_data.data["point_idx"] == target_index]
        vel_arr = point_data[component].to_numpy()
        time_arr = point_data["time_step"].to_numpy()

        spec_dens, norm_freq = spectral_density_function(
            velocity_signal=vel_arr,
            timestamps=time_arr,
            reference_velocity=normalization_params.reference_velocity,
            characteristic_length=normalization_params.characteristic_length,
        )
        spectral_data[f"S ({component})"] = spec_dens
        spectral_data[f"f ({component})"] = norm_freq

    return spectral_data


def calculate_autocorrelation(
    inflow_data: InflowData, anchor_point_idx: int, for_components: list[VelocityComponents]
) -> pd.DataFrame:
    """Calculates the autocorrelation from an anchor point

    Args:
        inflow_data (InflowData): Inflow data structure containing points and hist series
        anchor_point_idx (int): Index of the anchor point
        for_components (list[VelocityComponents]): List of components to calculate turbulence intensity from

    Returns:
        pd.DataFrame: Autocorrelation data with columns for each for_components preffixed by coef_*
    """
    anchor_data = inflow_data.data.loc[inflow_data.data["point_idx"] == anchor_point_idx].copy()
    anchor_data = anchor_data[for_components + ["point_idx", "time_step"]]
    for component in for_components:
        anchor_data[f"{component}_a"] = anchor_data[f"{component}"]
        anchor_data[f"{component}_a^2"] = anchor_data[f"{component}"] ** 2
    data_to_merge = anchor_data[
        ["time_step"]
        + [f"{component}_a{symbol}" for component in for_components for symbol in ["^2", ""]]
    ]
    merged_data = pd.merge(inflow_data.data, data_to_merge, on="time_step", how="left")
    for component in for_components:
        merged_data[f"{component}_{component}_a"] = (
            merged_data[f"{component}"] * merged_data[f"{component}_a"]
        )
    avg_data = merged_data.groupby("point_idx").mean()
    for component in for_components:
        avg_data[f"coef_{component}"] = (
            avg_data[f"{component}_{component}_a"]
            - avg_data[f"{component}"] * avg_data[f"{component}_a"]
        ) / (avg_data[f"{component}_a^2"] - avg_data[f"{component}_a"] ** 2)
    autocorrelation = avg_data[[f"coef_{c}" for c in for_components]].reset_index()
    return autocorrelation
