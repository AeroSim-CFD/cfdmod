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
