"""Streaming statistics computation from H5 timeseries files.

Replaces calculate_statistics_for_groups() from the old chunking pipeline.
Used primarily for Cp statistics where n_tri can be large.
"""

from __future__ import annotations

__all__ = ["calculate_statistics_from_h5"]

import pathlib

import h5py
import numpy as np
import pandas as pd

from cfdmod.io.xdmf import filter_keys_by_range, get_pressure_keys
from cfdmod.pressure.functions import (
    calculate_statistics,
    gumbel_extreme_values,
    moving_average_extreme_values,
    peak_extreme_values,
)
from cfdmod.pressure.parameters import (
    BasicStatisticModel,
    ParameterizedStatisticModel,
)


def calculate_statistics_from_h5(
    h5_path: pathlib.Path,
    group: str,
    statistics: list[BasicStatisticModel | ParameterizedStatisticModel],
    timestep_range: tuple[float, float] | None = None,
) -> pd.DataFrame:
    """Compute statistics over a timeseries H5 group.

    Basic stats (mean, rms, skewness, kurtosis) use a single-pass online
    algorithm (Welford) requiring O(n_points) memory.

    Parameterized stats needing the full dataset (Gumbel, Moving Average, Peak,
    Absolute) load data as a [n_time, n_points] array and apply the method.

    Args:
        h5_path (pathlib.Path): Timeseries H5 file
        group (str): Dataset group (e.g. "cp")
        statistics: List of statistics to compute
        timestep_range (tuple | None): Optional (t_min, t_max) filter

    Returns:
        pd.DataFrame: Statistics with stat names as columns, indexed by point
    """
    keys = get_pressure_keys(h5_path, group)
    if timestep_range is not None:
        keys = filter_keys_by_range(keys, timestep_range)

    if not keys:
        raise ValueError(f"No keys found in {h5_path}:{group} for the given range")

    stats_list = statistics
    statistics_names = [s.stats for s in stats_list]

    # Determine which stats can be done streaming vs need full data
    _STREAMING = {"mean", "rms", "skewness", "kurtosis"}
    _needs_full = False
    for s in stats_list:
        if s.stats in ("min", "max", "mean_eq") and hasattr(s, "params"):
            method = s.params.method_type  # type: ignore
            if method in ("Gumbel", "Moving Average", "Absolute", "Peak"):
                _needs_full = True
                break
        if s.stats in ("min", "max"):
            _needs_full = True

    # If all stats are streaming-compatible and no min/max, do single pass
    if not _needs_full and all(s in _STREAMING for s in statistics_names):
        return _streaming_only(h5_path, group, keys, stats_list)

    # Otherwise, load full data and call calculate_statistics
    return _full_load(h5_path, group, keys, stats_list)


def _streaming_only(
    h5_path: pathlib.Path,
    group: str,
    keys: list[tuple[float, str]],
    statistics: list[BasicStatisticModel | ParameterizedStatisticModel],
) -> pd.DataFrame:
    """Single-pass streaming statistics using Welford's algorithm."""
    n_steps = len(keys)
    n = 0
    mean_acc: np.ndarray | None = None
    M2: np.ndarray | None = None
    M3: np.ndarray | None = None
    M4: np.ndarray | None = None

    stats_names = [s.stats for s in statistics]

    with h5py.File(h5_path, "r") as f:
        grp = f[group]
        for _, t_key in keys:
            x = grp[t_key][:].astype(np.float64)
            n += 1
            if mean_acc is None:
                mean_acc = np.zeros_like(x)
                M2 = np.zeros_like(x)
                M3 = np.zeros_like(x)
                M4 = np.zeros_like(x)

            delta = x - mean_acc
            mean_acc += delta / n
            delta2 = x - mean_acc
            M2 += delta * delta2  # type: ignore
            M3 += delta * delta2**2  # type: ignore
            M4 += delta**2 * delta2**2  # type: ignore

    stats_df_dict: dict[str, np.ndarray] = {}
    if "mean" in stats_names:
        stats_df_dict["mean"] = mean_acc  # type: ignore
    if "rms" in stats_names:
        variance = M2 / (n - 1) if n > 1 else M2  # type: ignore
        stats_df_dict["rms"] = np.sqrt(variance)
    if "skewness" in stats_names:
        variance = M2 / n  # type: ignore
        with np.errstate(divide="ignore", invalid="ignore"):
            skew = np.where(variance > 0, M3 / (n * variance**1.5), 0.0)  # type: ignore
        stats_df_dict["skewness"] = skew
    if "kurtosis" in stats_names:
        variance = M2 / n  # type: ignore
        with np.errstate(divide="ignore", invalid="ignore"):
            kurt = np.where(variance > 0, M4 / (n * variance**2) - 3.0, 0.0)  # type: ignore
        stats_df_dict["kurtosis"] = kurt

    return pd.DataFrame(stats_df_dict)


def _full_load(
    h5_path: pathlib.Path,
    group: str,
    keys: list[tuple[float, str]],
    statistics: list[BasicStatisticModel | ParameterizedStatisticModel],
) -> pd.DataFrame:
    """Load all data as [n_time, n_points] and compute statistics."""
    with h5py.File(h5_path, "r") as f:
        grp = f[group]
        arrays = [grp[t_key][:].astype(np.float64) for _, t_key in keys]

    full_data = np.stack(arrays)
    n_points = full_data.shape[1]

    # Build a DataFrame compatible with calculate_statistics
    time_normalized = np.arange(len(keys), dtype=np.float64)
    df = pd.DataFrame(
        full_data, columns=[str(i) for i in range(n_points)]
    )
    df["time_normalized"] = time_normalized

    return calculate_statistics(df, statistics_to_apply=statistics)
