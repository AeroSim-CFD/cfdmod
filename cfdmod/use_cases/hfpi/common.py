from __future__ import annotations

from collections import defaultdict
from typing import Literal

import numpy as np
import pandas as pd


def get_moments_from_force(force: dict[str, np.ndarray], floor_heights: np.ndarray):
    moments = {}
    # Force in X causes -Y moment (right hand rule)
    moments["x"] = -force["y"] * floor_heights
    # Force in Y causes +Y moment (right hand rule)
    moments["y"] = force["x"] * floor_heights
    # Z is already a moment
    moments["z"] = force["z"].copy()
    return moments


def fill_forces_floors(forces_df: pd.DataFrame, n_floors: int):
    """Fill missing floors with zeros"""
    floors = [int(k) for k in forces_df if not isinstance(k, str) or k.isnumeric()]
    for i in range(n_floors):
        if i in floors:
            continue
        forces_df[str(i)] = 0

def series_cross_product(arm: np.ndarray, vx: pd.DataFrame|np.ndarray, vy: pd.DataFrame|np.ndarray) -> pd.DataFrame:
    return arm[0]*vy - arm[1]*vx

def get_stats_dct(
    dct: dict[str, np.ndarray], stats_type: Literal["min", "max", "mean"]
) -> dict[str, np.ndarray] | dict[str, float]:
    if stats_type == "max":
        return {k: v.max(axis=0) for k, v in dct.items()}
    elif stats_type == "min":
        return {k: v.min(axis=0) for k, v in dct.items()}
    elif stats_type == "mean":
        return {k: v.mean(axis=0) for k, v in dct.items()}
    raise ValueError(f"Invalid stats type: {stats_type!r}, supports only 'min', 'max', 'mean'")


def get_stats_dct_gumbell(
    dct: dict[str, np.ndarray], stats_type: Literal["min", "max", "mean"], dt:float|None=None
) -> dict[str, np.ndarray] | dict[str, float]:
    if stats_type in ["max","min"]:
        return {
            k: gumbel_extreme_value(
                hist_series=v,
                dt=dt,
                peak_duration=3,
                event_duration=10*60,
                extreme_type=stats_type,
                n_subdivisions=10,
                non_exceedance_probability=0.78,
            )
            for k, v in dct.items()
        }
    elif stats_type == "mean":
        return {k: v.mean(axis=0) for k, v in dct.items()}
    raise ValueError(f"Invalid stats type: {stats_type!r}, supports only 'min', 'max', 'mean'")

def get_stats_dct_peak_factor(
    dct: dict[str, np.ndarray], stats_type: Literal["min", "max", "mean"], peak_factor:float,
) -> dict[str, np.ndarray] | dict[str, float]:

    resp = {}
    for k, v in dct.items():
        mn = v.mean(axis=0)
        rms = (v - mn).std(axis=0)
        if stats_type == "max":
            resp[k] = mn + rms*peak_factor
        elif stats_type == "min":
            resp[k] = mn - rms*peak_factor
        elif stats_type == "mean":
            resp[k] = mn
        else:
            raise ValueError(f"Invalid stats type: {stats_type!r}, supports only 'min', 'max', 'mean'")
    return resp

def get_stats_among_dct(
    lst_dct: list[dict[str, np.ndarray] | dict[str, float]],
    stats_type: Literal["min", "max", "mean"],
) -> dict[str, np.ndarray] | dict[str, float]:
    if len(lst_dct) == 0:
        return {}
    if stats_type not in ("min", "max", "mean"):
        raise ValueError(f"Invalid stats type: {stats_type!r}, supports only 'min', 'max', 'mean'")
    keys = lst_dct[0].keys()
    dct: dict[str, np.ndarray] | dict[str, float] = {}
    for k in keys:
        vals = []
        for d in lst_dct:
            vals.append(d[k])
        arr = np.array(vals)
        func = arr.min if stats_type == "min" else (arr.max if stats_type == "max" else arr.mean)
        dct[k] = func(axis=0)
    return dct


def get_global_dct(dct: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    d = {k: v.sum(axis=1) for k, v in dct.items()}
    return d


def get_global_stats_dct_float(
    dcts: list[dict[str, float]], stats_type: Literal["min", "max", "mean"]
) -> dict[str, float]:
    grouped: dict[str, list[float]] = defaultdict(list)

    for d in dcts:
        for k, v in d.items():
            grouped[k].append(v)

    result: dict[str, float] = {}
    for k, values in grouped.items():
        if stats_type == "min":
            result[k] = min(values)
        elif stats_type == "max":
            result[k] = max(values)
        elif stats_type == "mean":
            result[k] = sum(values) / len(values)
        else:
            raise ValueError(
                f"Invalid stats_type: {stats_type!r}. Must be 'min', 'max', or 'mean'."
            )

    return result


def rotate_values_xy(values_proj: dict[str, np.ndarray], angle_rot: float):
    cos_theta = np.cos(np.radians(angle_rot))
    sin_theta = np.sin(np.radians(angle_rot))

    along_wind = cos_theta * values_proj["x"] - sin_theta * values_proj["y"]
    across_wind = sin_theta * values_proj["x"] + cos_theta * values_proj["y"]
    values_proj["x"] = along_wind
    values_proj["y"] = across_wind


def get_building_angle_rotate_across_along_wind(wind_direction: float, building_rotation: float):
    return (wind_direction - building_rotation + 90 + 360) % 360


def first_derivative(series: dict[str, np.ndarray], dt: float) -> dict[str, np.ndarray]:
    velocity = {}
    for axis in series:
        disp = disp_full[axis]
        v = np.zeros_like(disp, dtype=np.float32)
        # backward diference for internal points
        v[1:] = (disp[1:] - disp[:-1]) / dt
        # Forward difference for first point
        v[0] = (disp[1]- disp[0]) / dt
        velocity[axis] = v
    return velocity

def second_derivative(series: dict[str, np.ndarray], dt: float) -> dict[str, np.ndarray]:
    acceleration = {}
    for axis in series:
        disp = series[axis]
        acc = np.zeros_like(disp, dtype=np.float32)
        # Central difference for internal points
        acc[1:-1] = (disp[2:] - 2 * disp[1:-1] + disp[:-2]) / dt**2
        # Forward/backward difference for boundaries
        acc[0] = (disp[2] - 2 * disp[1] + disp[0]) / dt**2
        acc[-1] = (disp[-1] - 2 * disp[-2] + disp[-3]) / dt**2
        acceleration[axis] = acc
    return acceleration

def fit_gumbel_model(
    data: np.ndarray, 
    event_duration: float, 
    non_exceedance_probability: float
) -> float:
    """Fits the Gumbel model to predict extreme events

    Args:
        data (np.ndarray): Historic series
        params (ExtremeGumbelParamsModel): Parameters for Gumbel model analysis
        sample_duration (float): Duration of the sample

    Returns:
        float: Gumbel value for data
    """
    N = len(data)
    yR = -np.log(-np.log(non_exceedance_probability))
    y = [-np.log(-np.log(i / (N + 1))) for i in range(1, N + 1)]
    A = np.vstack([y, np.ones(len(y))]).T
    a_inv, U_T0 = np.linalg.lstsq(A, data, rcond=None)[0]
    U_T1 = U_T0 + a_inv * np.log(event_duration / (event_duration / N))
    extreme_val = a_inv * yR + U_T1  # This is the design value

    return extreme_val

def gumbel_extreme_value(
    hist_series: np.ndarray,
    dt: float,
    peak_duration: float,
    event_duration: float,
    extreme_type: Literal['min','max'],
    n_subdivisions: int = 10,
    non_exceedance_probability: float = 0.78,
) -> tuple[float, float]:
    """Apply extreme values analysis to coefficient historic series

    Args:
        params (ExtremeGumbelParamsModel): Parameters for extreme values calculation
        time_scale_factor (float): Value for converting time scales
        timestep_arr (np.ndarray): Array of simulated timesteps
        hist_series (np.ndarray): Coefficient historic series

    Returns:
        tuple[float, float]: Tuple with (min, max) extreme values
    """

    T0 = event_duration
    window_size = max(int(peak_duration / dt), 1)
    smooth_parent_cp = np.convolve(hist_series, np.ones(window_size) / window_size, mode="valid")

    sub_arrays = np.array_split(smooth_parent_cp, n_subdivisions)

    if extreme_type == "max":
        v_peak = np.array([np.max(sub_arr) for sub_arr in sub_arrays])
        v_peak = np.sort(v_peak)
    
    if extreme_type == "min":
        v_peak = np.array([np.min(sub_arr) for sub_arr in sub_arrays])
        v_peak = np.sort(v_peak)[::-1]

    peak_extreme_val = fit_gumbel_model(    
        data=v_peak, 
        event_duration=event_duration, 
        event_duration=event_duration,
        non_exceedance_probability=non_exceedance_probability
    )

    # It may return NaN values if the time series is invalid or has very few points
    peak_extreme_val = 0 if np.isnan(peak_extreme_val) else peak_extreme_val

    return peak_extreme_val