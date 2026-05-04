"""H5-backed wrapper for statistics computation.

File-in flow over cfdmod's standard timeseries H5 layout. Loads the
requested group (optionally restricted to a time range) and returns a
DataFrame indexed by feature with one column per statistic.

Picks a single-pass streaming path (Welford's algorithm) when all
requested stats are basic moments (mean / rms / skewness / kurtosis);
otherwise loads the whole timeseries into memory and delegates to
:func:`cfdmod.statistics.calculate_statistics`.
"""

from __future__ import annotations

__all__ = ["apply_statistics_h5"]

import pathlib

import h5py
import numpy as np
import pandas as pd

from cfdmod.io.xdmf import filter_keys_by_range, get_pressure_keys
from cfdmod.statistics.core import calculate_statistics
from cfdmod.statistics.specs import (
    BasicStatisticModel,
    ParameterizedStatisticModel,
)


_STREAMING = {"mean", "rms", "skewness", "kurtosis"}


def apply_statistics_h5(
    h5_path: pathlib.Path,
    group: str,
    statistics: list[BasicStatisticModel | ParameterizedStatisticModel],
    timestep_range: tuple[float, float] | None = None,
) -> pd.DataFrame:
    """Compute statistics over a timeseries H5 group.

    Args:
        h5_path: Timeseries H5 file (cfdmod layout).
        group: Group name inside the H5 (e.g. ``"cp"``, ``"cf_x"``).
        statistics: List of statistics to compute.
        timestep_range: Optional ``(t_min, t_max)`` filter applied before
            computing.

    Returns:
        DataFrame indexed by feature/point (rows), one column per statistic.
    """
    keys = get_pressure_keys(h5_path, group)
    if timestep_range is not None:
        keys = filter_keys_by_range(keys, timestep_range)
    if not keys:
        raise ValueError(f"No keys found in {h5_path}:{group} for the given range")

    statistics_names = [s.stats for s in statistics]
    needs_full = any(s.stats in ("min", "max", "mean_eq") for s in statistics)

    if not needs_full and all(s in _STREAMING for s in statistics_names):
        return _streaming_only(h5_path, group, keys, statistics)
    return _full_load(h5_path, group, keys, statistics)


def _streaming_only(
    h5_path: pathlib.Path,
    group: str,
    keys: list[tuple[float, str]],
    statistics: list[BasicStatisticModel | ParameterizedStatisticModel],
) -> pd.DataFrame:
    """Single-pass streaming statistics using Welford's algorithm."""
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

    out: dict[str, np.ndarray] = {}
    if "mean" in stats_names:
        out["mean"] = mean_acc  # type: ignore
    if "rms" in stats_names:
        variance = M2 / (n - 1) if n > 1 else M2  # type: ignore
        out["rms"] = np.sqrt(variance)
    if "skewness" in stats_names:
        variance = M2 / n  # type: ignore
        with np.errstate(divide="ignore", invalid="ignore"):
            out["skewness"] = np.where(
                variance > 0, M3 / (n * variance**1.5), 0.0  # type: ignore
            )
    if "kurtosis" in stats_names:
        variance = M2 / n  # type: ignore
        with np.errstate(divide="ignore", invalid="ignore"):
            out["kurtosis"] = np.where(
                variance > 0, M4 / (n * variance**2) - 3.0, 0.0  # type: ignore
            )
    return pd.DataFrame(out)


def _full_load(
    h5_path: pathlib.Path,
    group: str,
    keys: list[tuple[float, str]],
    statistics: list[BasicStatisticModel | ParameterizedStatisticModel],
) -> pd.DataFrame:
    """Load the full series and delegate to calculate_statistics."""
    with h5py.File(h5_path, "r") as f:
        grp = f[group]
        arrays = [grp[t_key][:].astype(np.float64) for _, t_key in keys]
    full_data = np.stack(arrays)
    n_points = full_data.shape[1]

    df = pd.DataFrame(full_data, columns=[str(i) for i in range(n_points)])
    df["time_normalized"] = np.arange(len(keys), dtype=np.float64)
    return calculate_statistics(df, statistics_to_apply=statistics)
