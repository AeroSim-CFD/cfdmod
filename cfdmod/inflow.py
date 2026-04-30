"""Inflow profile analysis: read solver inflow timeseries and compute
mean velocity, turbulence intensity, spectral density, and
autocorrelation.

The new layout is XDMF+H5 (one h5py group per velocity component with
``/t{T}`` per-timestep arrays plus ``/meta/time_steps``); a legacy
pandas-HDFStore reader is still here behind a ``DeprecationWarning``.
"""

from __future__ import annotations

import pathlib
import warnings
from dataclasses import dataclass
from typing import Literal

import h5py
import numpy as np
import pandas as pd
import scipy
from scipy.ndimage import gaussian_filter

__all__ = [
    "VelocityComponents",
    "NormalizationParameters",
    "InflowData",
    "spectral_density_function",
    "calculate_mean_velocity",
    "calculate_turbulence_intensity",
    "calculate_spectral_density",
    "calculate_autocorrelation",
]


VelocityComponents = Literal["ux", "uy", "uz"]


# ---------------------------------------------------------------------------
# Data containers
# ---------------------------------------------------------------------------


@dataclass
class NormalizationParameters:
    reference_velocity: float
    characteristic_length: float


def _read_inflow_h5(hist_series_path: pathlib.Path) -> pd.DataFrame:
    """Read an inflow timeseries H5, accepting both legacy and new layouts.

    New layout: h5py datasets under ``/{component}/t{T}`` (per-component,
    per-timestep arrays of shape ``(n_points,)``) plus ``/meta/time_steps``.

    Legacy layout: pandas HDFStore with multiple group keys, each holding a
    DataFrame fragment (columns include ``time_step``, ``point_idx``,
    ``ux``, ``uy``, ``uz``). Reading the legacy layout emits a
    ``DeprecationWarning``.
    """
    with h5py.File(hist_series_path, "r") as f:
        top_keys = set(f.keys())
        velocity_groups = [k for k in ("ux", "uy", "uz") if k in top_keys]
        if velocity_groups and "meta" in top_keys:
            time_steps = f["meta"]["time_steps"][:]
            frames = []
            for comp in velocity_groups:
                grp = f[comp]
                step_keys = sorted(grp.keys(), key=lambda k: float(k[1:]))
                arrays = np.stack([grp[k][:].astype(np.float64) for k in step_keys])
                n_points = arrays.shape[1]
                df = pd.DataFrame(
                    {
                        "time_step": np.repeat(time_steps[: len(step_keys)], n_points),
                        "point_idx": np.tile(np.arange(n_points), len(step_keys)),
                        comp: arrays.reshape(-1),
                    }
                )
                frames.append(df.set_index(["time_step", "point_idx"]))
            return pd.concat(frames, axis=1).reset_index()

    warnings.warn(
        f"Reading legacy pandas-HDFStore inflow file {hist_series_path}. "
        "Convert to the new XDMF+H5 layout (one h5py group per velocity "
        "component with /t{T} datasets and /meta/time_steps); legacy support "
        "will be removed in a future release.",
        DeprecationWarning,
        stacklevel=2,
    )
    data_dfs = []
    with pd.HDFStore(hist_series_path, mode="r") as data_store:
        for key in data_store.keys():
            data_dfs.append(data_store.get(key))
    data = pd.concat(data_dfs)
    data.sort_values(
        by=[c for c in ["time_step", "point_idx"] if c in data.columns],
        inplace=True,
    )
    return data


class InflowData:
    def __init__(self, data: pd.DataFrame, points: pd.DataFrame):
        self.data = data
        self.points = points

    @classmethod
    def from_files(
        cls,
        hist_series_path: pathlib.Path,
        points_path: pathlib.Path,
    ) -> InflowData:
        """Reads data from file and builds an ``InflowData``.

        The inflow data DataFrame must contain the columns ``(ux, uy, uz)``.
        If any are missing, downstream calculations on those components are
        skipped; the remaining components still work.

        Args:
            hist_series_path: Path of the historic series (point_idx,
                ux, uy, uz velocities). ``.h5`` (XDMF+H5 or legacy
                HDFStore) or ``.csv``.
            points_path: Path of the points information (idx, x, y, z
                coordinates). CSV.

        Returns:
            InflowData: Inflow data object.
        """
        hist_series_format = hist_series_path.name.split(".")[-1]
        if hist_series_format == "csv":
            data = pd.read_csv(hist_series_path)
        elif hist_series_format == "h5":
            data = _read_inflow_h5(hist_series_path)
        else:
            raise Exception(
                f"Extension {hist_series_format} not supported for hist series!"
            )
        points = pd.read_csv(points_path)
        return InflowData(data=data, points=points)


# ---------------------------------------------------------------------------
# Analysis functions
# ---------------------------------------------------------------------------


def spectral_density_function(
    velocity_signal: np.ndarray,
    timestamps: np.ndarray,
    reference_velocity: float,
    characteristic_length: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Perform an FFT over a velocity signal.

    Args:
        velocity_signal: Array of instantaneous velocity signal.
        timestamps: Array of timestamps of the signal.
        reference_velocity: Reference velocity used for normalisation.
        characteristic_length: Characteristic length used for normalisation.

    Returns:
        ``(normalized_frequency, spectral_density)`` arrays.
    """

    def filter_avg_data(data: np.ndarray) -> np.ndarray:
        return gaussian_filter(data, sigma=3)  # sigma smooths the curve

    delta_t = timestamps[1] - timestamps[0]
    signal_frequency = 1 / delta_t

    xf, yf = scipy.signal.periodogram(velocity_signal, signal_frequency, scaling="density")
    st = np.std(velocity_signal)
    yf = xf * yf / st**2
    xf = xf * characteristic_length / reference_velocity  # Strouhal number N = f * L / U

    yf = filter_avg_data(yf)
    return xf[2:], yf[2:]


def calculate_mean_velocity(
    inflow_data: InflowData,
    for_components: list[VelocityComponents],
) -> pd.DataFrame:
    """Calculate the per-point time-mean of each requested velocity component.

    Args:
        inflow_data: Inflow data structure containing points and hist series.
        for_components: Components to calculate mean velocity for.

    Returns:
        DataFrame with one column per ``for_components`` entry suffixed
        by ``_mean``.
    """
    if not all(c in inflow_data.data.columns for c in for_components + ["point_idx"]):
        raise ValueError("Components must be inside inflow profile data columns")

    group_by_point_idx = inflow_data.data.groupby("point_idx")
    velocity_data = group_by_point_idx.agg(
        {component: "mean" for component in for_components}
    ).reset_index()
    velocity_data.columns = [
        col + "_mean" if col in for_components else col for col in velocity_data.columns
    ]
    return velocity_data


def calculate_turbulence_intensity(
    inflow_data: InflowData,
    for_components: list[VelocityComponents],
) -> pd.DataFrame:
    """Calculate the per-point turbulence intensity for each requested component.

    Args:
        inflow_data: Inflow data structure containing points and hist series.
        for_components: Components to calculate turbulence intensity for.

    Returns:
        DataFrame with one column per ``for_components`` entry prefixed by ``I_``.
    """
    if not all(c in inflow_data.data.columns for c in for_components + ["point_idx"]):
        raise ValueError("Components must be inside inflow profile data columns")

    group_by_point_idx = inflow_data.data.groupby("point_idx")
    turbulence_data = group_by_point_idx.agg(
        {component: ["mean", "std"] for component in for_components}
    ).reset_index()
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
    """Compute the spectral density for a given target point index.

    Args:
        inflow_data: Inflow data structure containing points and hist series.
        target_index: Index of the target point.
        for_components: Components to compute spectral density for.
        normalization_params: Parameters for spectral density normalisation.

    Returns:
        DataFrame with columns ``S (component)`` and ``f (component)`` per
        component in ``for_components``.
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
    inflow_data: InflowData,
    anchor_point_idx: int,
    for_components: list[VelocityComponents],
) -> pd.DataFrame:
    """Calculate the autocorrelation between each point and an anchor point.

    Args:
        inflow_data: Inflow data structure containing points and hist series.
        anchor_point_idx: Index of the anchor point.
        for_components: Components to calculate autocorrelation for.

    Returns:
        DataFrame with one column per ``for_components`` entry prefixed by ``coef_``.
    """
    anchor_data = inflow_data.data.loc[inflow_data.data["point_idx"] == anchor_point_idx].copy()
    anchor_data = anchor_data[for_components + ["point_idx", "time_step"]]
    for component in for_components:
        anchor_data[f"{component}_a"] = anchor_data[component]
        anchor_data[f"{component}_a^2"] = anchor_data[component] ** 2
    data_to_merge = anchor_data[
        ["time_step"]
        + [f"{component}_a{symbol}" for component in for_components for symbol in ["^2", ""]]
    ]
    merged_data = pd.merge(inflow_data.data, data_to_merge, on="time_step", how="left")
    for component in for_components:
        merged_data[f"{component}_{component}_a"] = (
            merged_data[component] * merged_data[f"{component}_a"]
        )
    avg_data = merged_data.groupby("point_idx").mean()
    for component in for_components:
        avg_data[f"coef_{component}"] = (
            avg_data[f"{component}_{component}_a"]
            - avg_data[component] * avg_data[f"{component}_a"]
        ) / (avg_data[f"{component}_a^2"] - avg_data[f"{component}_a"] ** 2)
    autocorrelation = avg_data[[f"coef_{c}" for c in for_components]].reset_index()
    return autocorrelation
