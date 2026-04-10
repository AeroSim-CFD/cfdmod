from __future__ import annotations

from collections import defaultdict
from typing import Literal

import numpy as np
import pandas as pd
from scipy.signal import convolve


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


def series_cross_product(
    arm: np.ndarray, vx: pd.DataFrame | np.ndarray, vy: pd.DataFrame | np.ndarray
) -> pd.DataFrame:
    return arm[0] * vy - arm[1] * vx


def move_loads_ref_from_CM_to_origin(
    forces: dict[str, np.ndarray],
    moments: dict[str, np.ndarray],
    cm_positions: pd.DataFrame,
) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray]]:
    """Transforms forces and moments of force from the coordinate system of center of mass to original coordinate system.

    Args:
        forces dict[str, np.ndarray]: dictionary with force. Keys are ['x','y''z'] and items are numpy arrays N timesteps by F floors.
        moments dict[str, np.ndarray]: dictionary with moments of force. Keys are ['x','y''z'] and items are numpy arrays N timesteps by F floors.
        structural_data (HFPIStructuralData): object with structural data

    Returns:
        tuple[dict[str, np.ndarray],dict[str, np.ndarray]]: Transformed dictionaries of force and moment of force in that order
    """
    fx, fy, mz = forces["x"], forces["y"], moments["z"]
    new_mz = mz.copy()
    n_floors = fx.shape[1]
    for n_floor in range(n_floors):
        CM_pos = np.array((cm_positions.iloc[n_floor][["XR", "YR"]]))
        new_mz[:, n_floor] = mz[:, n_floor] + series_cross_product(
            CM_pos, fx[:, n_floor], fy[:, n_floor]
        )
    new_moments = {**moments, "z": new_mz}
    return forces, new_moments


def move_loads_ref_from_origin_to_CM(
    forces: dict[str, np.ndarray],
    moments: dict[str, np.ndarray],
    cm_positions: pd.DataFrame,
) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray]]:
    """Transforms forces and moments of force from the coordinate system of center of mass to original coordinate system.

    Args:
        forces dict[str, np.ndarray]: dictionary with force. Keys are ['x','y''z'] and items are numpy arrays N timesteps by F floors.
        moments dict[str, np.ndarray]: dictionary with moments of force. Keys are ['x','y''z'] and items are numpy arrays N timesteps by F floors.
        structural_data (HFPIStructuralData): object with structural data

    Returns:
        tuple[dict[str, np.ndarray],dict[str, np.ndarray]]: Transformed dictionaries of force and moment of force in that order
    """
    cm_positions_inv = cm_positions.copy()
    cm_positions_inv[["XR", "YR"]] = -cm_positions_inv[["XR", "YR"]]
    return move_loads_ref_from_CM_to_origin(
        forces,
        moments,
        cm_positions_inv,
    )


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
    dct: dict[str, np.ndarray], stats_type: Literal["min", "max", "mean"], dt: float | None = None
) -> dict[str, np.ndarray] | dict[str, float]:
    if stats_type in ["max", "min"]:
        return {
            k: gumbel_extreme_value(
                hist_series=v,
                dt=dt,
                peak_duration=3,
                event_duration=10 * 60,
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
    dct: dict[str, np.ndarray],
    stats_type: Literal["min", "max", "mean"],
    peak_factor: float,
) -> dict[str, np.ndarray] | dict[str, float]:

    resp = {}
    for k, v in dct.items():
        mn = v.mean(axis=0)
        rms = (v - mn).std(axis=0)
        if stats_type == "max":
            resp[k] = mn + rms * peak_factor
        elif stats_type == "min":
            resp[k] = mn - rms * peak_factor
        elif stats_type == "mean":
            resp[k] = mn
        else:
            raise ValueError(
                f"Invalid stats type: {stats_type!r}, supports only 'min', 'max', 'mean'"
            )
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
        v[0] = (disp[1] - disp[0]) / dt
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


def moving_filter(hist_series: np.ndarray, dt: float, peak_duration: float) -> np.ndarray:
    window_size = max(int(peak_duration / dt), 1)
    kernel = np.ones(window_size) / window_size
    smooth_parent_cp = convolve(hist_series, kernel, mode="valid")
    return smooth_parent_cp


def reescale_event_duration_peak(
    loc: float,
    scale: float,
    original_time: float,
    new_time: float,
    extreme_type: Literal["min", "max"],
) -> tuple[float, float]:

    sign = 1 if extreme_type == "max" else -1
    new_scale = scale
    new_loc = loc + sign * scale * np.log(new_time / original_time)
    return new_loc, new_scale


def gumbel_extreme_value(
    hist_series: np.ndarray,
    dt: float,
    peak_duration: float,
    event_duration: float,
    extreme_type: Literal["min", "max"],
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

    if hist_series.ndim > 1:
        raise ValueError("Gumbel fit works only on 1D arrays")

    smoothed_parent = moving_filter(hist_series, dt, peak_duration)

    sub_arrays = np.array_split(smoothed_parent, n_subdivisions)
    orig_time_duration = len(hist_series) * dt / n_subdivisions

    if extreme_type == "max":
        v_peak = np.array([np.max(sub_arr, axis=0) for sub_arr in sub_arrays])
        from scipy.stats import gumbel_r

        loc, scale = gumbel_r.fit(v_peak)
        loc, scale = reescale_event_duration_peak(
            loc, scale, orig_time_duration, event_duration, extreme_type
        )
        p = non_exceedance_probability
        return gumbel_r.ppf(p, loc=loc, scale=scale)

    if extreme_type == "min":
        v_peak = np.array([np.min(sub_arr) for sub_arr in sub_arrays])
        from scipy.stats import gumbel_l

        loc, scale = gumbel_l.fit(v_peak)
        loc, scale = reescale_event_duration_peak(
            loc, scale, orig_time_duration, event_duration, extreme_type
        )
        p = 1 - non_exceedance_probability
        return gumbel_l.ppf(p, loc=loc, scale=scale)
