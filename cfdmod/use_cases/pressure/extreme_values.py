from __future__ import annotations

__all__ = ["ExtremeValuesParameters", "calculate_extreme_values"]


import math
from typing import Literal

import numpy as np
from pydantic import BaseModel, Field, model_validator

EXTREME_MODEL = Literal["Gumbell", "Moving average"]


class ExtremeValuesParameters(BaseModel):
    CST_real: float = Field(
        ..., title="CST real", description="Value for real scale Convective Scale Time"
    )
    CST_sim: float = Field(
        ..., title="CST simulated", description="Value for simulation scale Convective Scale Time"
    )
    extreme_model: EXTREME_MODEL = Field(
        ...,
        title="Extreme values model",
        description="Model to use for extreme values calculation",
    )
    parameters: dict = Field(
        ..., title="Extreme values parameters", description="Parameters for extreme values models"
    )
    time_scale_correction_factor: float = Field(
        0.61,
        title="Time scale factor",
        description="Correction factor for time scaling extreme events values",
    )

    @property
    def time_scale(self) -> float:
        return self.CST_real / self.CST_sim

    @model_validator(mode="after")
    def validate_params(self):
        if self.extreme_model == "Gumbell":
            for expected_param in ["t", "T0", "T1", "yR"]:
                if expected_param not in self.parameters.keys():
                    raise KeyError(
                        f"Extreme value model {self.extreme_model} requires {expected_param} as parameter"
                    )
        elif self.extreme_model == "Moving average":
            for expected_param in ["window_size_real"]:
                if expected_param not in self.parameters.keys():
                    raise KeyError(
                        f"Extreme value model {self.extreme_model} requires {expected_param} as parameter"
                    )
        else:
            raise Exception(f"Invalid model type {self.extreme_model} for extreme values")
        return self


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
    U_T1 = U_T0 + a_inv * math.log(params.parameters["T1"] / params.parameters["T0"])
    extreme_val = a_inv * params.parameters["yR"] + U_T1  # This is the design value

    return extreme_val


def gumbell_extreme_values(
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

    window_size = int(params.parameters["t"] / (time[1] - time[0]))
    smooth_parent_cp = np.convolve(hist_series, np.ones(window_size) / window_size, mode="valid")

    new_time = time[window_size // 2 - 2 : -window_size // 2 - 1]
    N = int(round((new_time[-1] - new_time[0]) / params.parameters["T0"]))  # num_divisions
    sub_arrays = np.array_split(smooth_parent_cp, N)

    cp_max = np.array([np.max(sub_arr) for sub_arr in sub_arrays])
    cp_min = np.array([np.min(sub_arr) for sub_arr in sub_arrays])

    cp_max = np.sort(cp_max)
    cp_min = np.sort(cp_min)[::-1]

    max_extreme_val = fit_gumbel_model(cp_max, params=params)
    min_extreme_val = fit_gumbel_model(cp_min, params=params)

    return min_extreme_val, max_extreme_val


def moving_average_extreme_values(
    params: ExtremeValuesParameters, hist_series: np.ndarray
) -> tuple[float, float]:
    """Apply extreme values analysis to coefficient historic series using moving average model

    Args:
        params (ExtremeValuesParameters): Parameters for extreme values calculation
        hist_series (np.ndarray): Coefficient historic series

    Returns:
        tuple[float, float]: Tuple with (min, max) extreme values
    """
    window_size = math.floor(params.parameters["window_size_real"] / params.time_scale)

    kernel = np.ones(window_size) / window_size
    smoothed_signal = np.convolve(hist_series, kernel, mode="same")

    min_extreme_val = smoothed_signal.min()
    max_extreme_val = smoothed_signal.max()

    return min_extreme_val, max_extreme_val
