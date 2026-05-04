"""Statistics computation core.

Two public surfaces:

- :func:`apply_statistics` -- pure numpy. Pass a ``(n_time,)`` or
  ``(n_time, n_features)`` array, the matching ``time`` axis, and a
  list of statistic specs; get back a DataFrame indexed by feature
  with one column per statistic. No file I/O; no DataFrame plumbing
  on the caller side. Use this from notebooks, custom pipelines, or
  any source that is not cfdmod's H5 layout.
- :func:`calculate_statistics` -- DataFrame entry. Same dispatch, but
  the caller hands in a DataFrame with a ``time_normalized`` column
  and one column per feature. Used internally by the pressure
  pipeline (Cf, Cm, Ce) and re-exported from
  :mod:`cfdmod.pressure.functions` for back-compat.

The H5-backed file flow lives in :mod:`cfdmod.statistics.h5`.
"""

from __future__ import annotations

__all__ = [
    "apply_statistics",
    "calculate_statistics",
    "calculate_extreme_values",
    "calculate_mean_equivalent",
    "extreme_values_analysis",
    "fit_gumbel_model",
    "gumbel_extreme_values",
    "peak_extreme_values",
]

import math

import numpy as np
import pandas as pd

from cfdmod.statistics.specs import (
    BasicStatisticModel,
    ExtremeGumbelParamsModel,
    ExtremePeakParamsModel,
    ParameterizedStatisticModel,
    StatisticsParamsModel,
)


def fit_gumbel_model(
    data: np.ndarray,
    params: ExtremeGumbelParamsModel,
    sample_duration: float,
) -> float:
    """Fit the Gumbel model to predict extreme events."""
    N = len(data)
    y = [-math.log(-math.log(i / (N + 1))) for i in range(1, N + 1)]
    A = np.vstack([y, np.ones(len(y))]).T
    a_inv, U_T0 = np.linalg.lstsq(A, data, rcond=None)[0]
    U_T1 = U_T0 + a_inv * math.log(params.event_duration / (sample_duration / N))
    return a_inv * params.yR + U_T1


def gumbel_extreme_values(
    params: ExtremeGumbelParamsModel,
    timestep_arr: np.ndarray,
    hist_series: np.ndarray,
) -> tuple[float, float]:
    """Apply Gumbel extreme values analysis to a coefficient historic series."""
    CST_full_scale = params.full_scale_characteristic_length / params.full_scale_U_H
    time = (timestep_arr - timestep_arr[0]) * CST_full_scale
    T0 = time[-1]
    window_size = max(int(params.peak_duration / (time[1] - time[0])), 1)
    smooth_parent_cp = np.convolve(
        hist_series, np.ones(window_size) / window_size, mode="valid"
    )
    sub_arrays = np.array_split(smooth_parent_cp, params.n_subdivisions)
    cp_max = np.sort(np.array([np.max(sub) for sub in sub_arrays]))
    cp_min = np.sort(np.array([np.min(sub) for sub in sub_arrays]))[::-1]
    max_val = fit_gumbel_model(cp_max, params=params, sample_duration=T0)
    min_val = fit_gumbel_model(cp_min, params=params, sample_duration=T0)
    min_val = 0 if np.isnan(min_val) else min_val
    max_val = 0 if np.isnan(max_val) else max_val
    return min_val, max_val


def peak_extreme_values(
    params: ExtremePeakParamsModel,
    hist_series: np.ndarray,
) -> tuple[float, float]:
    """Apply peak factor extreme values analysis."""
    avg = hist_series.mean()
    std = hist_series.std()
    return avg - params.peak_factor * std, avg + params.peak_factor * std


def extreme_values_analysis(
    params: StatisticsParamsModel,
    data_df: pd.DataFrame,
    timestep_arr: np.ndarray,
) -> pd.DataFrame:
    """Perform extreme-values analysis on a DataFrame column-by-column."""
    if params.method_type == "Absolute":
        return data_df.apply(lambda x: (x.min(), x.max()))
    if params.method_type == "Gumbel":
        return data_df.apply(
            lambda x: gumbel_extreme_values(
                params=params, timestep_arr=timestep_arr, hist_series=x
            )
        )
    if params.method_type == "Peak":
        return data_df.apply(lambda x: peak_extreme_values(params=params, hist_series=x))
    raise ValueError(f"Unknown method_type: {params.method_type}")


def calculate_extreme_values(
    extreme_statistics: list[ParameterizedStatisticModel],
    timestep_arr: np.ndarray,
    data_df: pd.DataFrame,
) -> dict[str, pd.Series]:
    """Calculate extreme values from historical data."""
    stats_df_dict: dict[str, pd.Series] = {}
    stats = [s for s in extreme_statistics if s.stats in ["min", "max"]]
    if (
        len(set([s.stats for s in stats])) == len(stats) == 2
        and len(set([s.params.method_type for s in stats])) == 1
    ):
        extremes_df = extreme_values_analysis(
            params=stats[0].params, data_df=data_df, timestep_arr=timestep_arr
        )
        stats_df_dict["min"] = extremes_df.iloc[0]
        stats_df_dict["max"] = extremes_df.iloc[1]
    else:
        for stat in stats:
            extremes_df = extreme_values_analysis(
                params=stat.params, data_df=data_df, timestep_arr=timestep_arr
            )
            target_index = 0 if stat.stats == "min" else 1
            stats_df_dict[stat.stats] = extremes_df.iloc[target_index]
    return stats_df_dict


def calculate_mean_equivalent(
    statistics_to_apply: list[BasicStatisticModel | ParameterizedStatisticModel],
    stats_df_dict: dict[str, pd.Series],
) -> np.ndarray:
    """Calculate mean-equivalent values from min/max/mean."""
    comparison_df = pd.DataFrame()
    mean_eq_stat = [s for s in statistics_to_apply if s.stats == "mean_eq"][0]
    scale_factor = mean_eq_stat.params.scale_factor
    for stat_lbl in ["min", "max", "mean"]:
        comparison_df[stat_lbl] = stats_df_dict[stat_lbl].copy()
        comparison_df[stat_lbl] *= 1 if stat_lbl == "mean" else scale_factor
    max_abs_col_index = np.abs(comparison_df.values).argmax(axis=1)
    return comparison_df.values[np.arange(len(comparison_df)), max_abs_col_index]


def calculate_statistics(
    historical_data: pd.DataFrame,
    statistics_to_apply: list[BasicStatisticModel | ParameterizedStatisticModel],
) -> pd.DataFrame:
    """Calculate statistics for a coefficient historic series.

    Args:
        historical_data: Matrix-form DataFrame with a ``time_normalized``
            column and one column per feature.
        statistics_to_apply: List of statistics to compute.

    Returns:
        DataFrame indexed by feature (rows), one column per statistic.
    """
    stats_df_dict: dict[str, pd.Series] = {}
    statistics_list = [s.stats for s in statistics_to_apply]
    data_df = historical_data.drop(columns=["time_normalized"])

    if "mean" in statistics_list:
        stats_df_dict["mean"] = data_df.mean()
    if "rms" in statistics_list:
        stats_df_dict["rms"] = data_df.std()
    if "skewness" in statistics_list:
        stats_df_dict["skewness"] = data_df.skew()
    if "kurtosis" in statistics_list:
        stats_df_dict["kurtosis"] = data_df.kurt()
    if "min" in statistics_list or "max" in statistics_list:
        stats = [s for s in statistics_to_apply if s.stats in ["min", "max"]]
        stats_df_dict = stats_df_dict | calculate_extreme_values(
            extreme_statistics=stats,
            timestep_arr=historical_data["time_normalized"].to_numpy(),
            data_df=data_df,
        )
    if "mean_eq" in statistics_list:
        stats_df_dict["mean_eq"] = calculate_mean_equivalent(
            statistics_to_apply=statistics_to_apply, stats_df_dict=stats_df_dict
        )

    return pd.DataFrame(stats_df_dict)


def apply_statistics(
    data: np.ndarray,
    *,
    time: np.ndarray,
    statistics: list[BasicStatisticModel | ParameterizedStatisticModel],
) -> pd.DataFrame:
    """Compute statistics for each column of ``data`` along axis 0.

    Pure-numpy entry point. Builds the DataFrame the dispatch below
    expects from your raw array + time axis, runs
    :func:`calculate_statistics`, and returns the result.

    Args:
        data: ``(n_time,)`` 1D or ``(n_time, n_features)`` 2D array.
            Statistics are computed per column.
        time: ``(n_time,)`` time axis. Required because Gumbel uses
            the sample duration; passed through verbatim to
            :func:`extreme_values_analysis`.
        statistics: List of statistic specs. Empty list raises.

    Returns:
        DataFrame indexed by feature (0..n_features-1), one column per
        statistic. For 1D input this is a single-row DataFrame.

    Raises:
        ValueError: ``statistics`` empty, ``data`` not 1D/2D, or
            ``time.size != data.shape[0]``.
    """
    if not statistics:
        raise ValueError("apply_statistics: statistics list is empty")

    arr = np.asarray(data)
    if arr.ndim == 1:
        arr = arr.reshape(-1, 1)
    if arr.ndim != 2:
        raise ValueError(
            f"apply_statistics: data must be 1D or 2D; got {arr.ndim}D shape {arr.shape}"
        )

    time_arr = np.asarray(time, dtype=np.float64)
    if time_arr.ndim != 1 or time_arr.size != arr.shape[0]:
        raise ValueError(
            f"apply_statistics: time must be 1D with length {arr.shape[0]}; "
            f"got shape {time_arr.shape}"
        )

    df = pd.DataFrame(arr, columns=[str(i) for i in range(arr.shape[1])])
    df["time_normalized"] = time_arr
    return calculate_statistics(df, statistics_to_apply=statistics)
