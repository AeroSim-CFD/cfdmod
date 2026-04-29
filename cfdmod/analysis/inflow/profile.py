from __future__ import annotations

import pathlib
import warnings
from dataclasses import dataclass

import h5py
import numpy as np
import pandas as pd


@dataclass
class NormalizationParameters:
    reference_velocity: float
    characteristic_length: float


def _read_inflow_h5(hist_series_path: pathlib.Path) -> pd.DataFrame:
    """Read an inflow timeseries H5, accepting both legacy and new layouts.

    New layout: h5py datasets under /{component}/t{T} (per-component, per-timestep
    arrays of shape (n_points,)) plus /meta/time_steps.

    Legacy layout: pandas HDFStore with multiple group keys, each holding a
    DataFrame fragment (columns include time_step, point_idx, ux, uy, uz).
    Reading the legacy layout emits a DeprecationWarning.
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
    def from_files(cls, hist_series_path: pathlib.Path, points_path: pathlib.Path) -> InflowData:
        """Reads data from file and builds a InflowData
        The inflow data dataframe must contain the columns (ux, uy, uz)
        If any are missing, it won't be able to perform calculations over the components that are missing,
        but will be able to perform calculations over the components that are present

        Args:
            hist_series_path (pathlib.Path): Path of the historic series (point_idx, ux, uy, uz velocities)
            points_path (pathlib.Path): Path of the points information (idx, x, y, z coordinates)

        Returns:
            InflowData: Inflow data object
        """
        hist_series_format = hist_series_path.name.split(".")[-1]
        if hist_series_format == "csv":
            data = pd.read_csv(hist_series_path)
        elif hist_series_format == "h5":
            data = _read_inflow_h5(hist_series_path)
        else:
            raise Exception(f"Extension {hist_series_format} not supported for hist series!")
        points = pd.read_csv(points_path)
        return InflowData(data=data, points=points)
