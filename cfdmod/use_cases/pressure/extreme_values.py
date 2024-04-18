from __future__ import annotations

__all__ = ["ExtremeValuesParameters", "calculate_extreme_values"]


import math

import numpy as np
from pydantic import BaseModel, Field


class ExtremeValuesParameters(BaseModel):
    CST_real: float = Field(
        ..., title="CST real", description="Value for real scale Convective Scale Time"
    )
    CST_sim: float = Field(
        ..., title="CST simulated", description="Value for simulation scale Convective Scale Time"
    )
    t: float = Field(..., title="Event duration", description="Extreme event duration time")
    T0: float = Field(
        ...,
        title="Actual observation period",
        description="Value for actual observation period",
    )
    T1: float = Field(
        ...,
        title="Target observation period",
        description="Value for target observation period",
    )
    yR: float = Field(
        1.4,
        title="Probabilistic parameter",
        description="Reduced parameter corresponding to 78 percent of non exceeding phenomenon",
    )
    time_scale_correction_factor: float = Field(
        0.61,
        title="Time scale factor",
        description="Correction factor for time scaling extreme events values",
    )

    @property
    def time_scale(self) -> float:
        return self.CST_real / self.CST_sim


def fit_gumbel_model(data: np.ndarray, params: ExtremeValuesParameters) -> float:
    """Fits the Gumbel model to predict extreme events

    Args:
        data (np.ndarray): Historic series
        params (ExtremeValuesParameters): Parameters for Gumbel model analysis

    Returns:
        float: Gumbel value for data
    """
    N = len(data)
    y = [-math.log(-math.log(i / (N + 1))) for i in range(1, N + 1)]
    A = np.vstack([y, np.ones(len(y))]).T
    a_inv, U_T0 = np.linalg.lstsq(A, data, rcond=None)[0]
    U_T1 = U_T0 + a_inv * math.log(params.T1 / params.T0)
    extreme_val = a_inv * params.yR + U_T1  # This is the design value

    return extreme_val


def calculate_extreme_values(
    params: ExtremeValuesParameters, timestep_arr: np.ndarray, hist_series: np.ndarray
) -> tuple[float, float]:
    """Apply extreme values analysis to coefficient historic series

    Args:
        params (ExtremeValuesParameters): Parameters for extreme values calculation
        timestep_arr (np.ndarray): Array of simulated timesteps
        hist_series (np.ndarray): Coefficient historic series

    Returns:
        tuple[float, float]: Tuple with (min, max) extreme values
    """
    time = (timestep_arr - timestep_arr[0]) * params.time_scale

    window_size = int(params.t / (time[1] - time[0]))
    smooth_parent_cp = np.convolve(hist_series, np.ones(window_size) / window_size, mode="valid")

    new_time = time[window_size // 2 - 2 : -window_size // 2 - 1]
    N = int(round((new_time[-1] - new_time[0]) / params.T0))  # num_divisions
    sub_arrays = np.array_split(smooth_parent_cp, N)

    cp_max = np.array([np.max(sub_arr) for sub_arr in sub_arrays])
    cp_min = np.array([np.min(sub_arr) for sub_arr in sub_arrays])

    cp_max = np.sort(cp_max)
    cp_min = np.sort(cp_min)[::-1]

    max_extreme_val = fit_gumbel_model(cp_max, params=params)
    min_extreme_val = fit_gumbel_model(cp_min, params=params)

    return min_extreme_val, max_extreme_val
