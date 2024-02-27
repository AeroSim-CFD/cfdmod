from __future__ import annotations

from dataclasses import dataclass
import pathlib

import pandas as pd


@dataclass
class NormalizationParameters:
    reference_velocity: float
    characteristic_length: float

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
            data_dfs = []
            with pd.HDFStore(hist_series_path, mode="r") as data_store:
                for key in data_store.keys():
                    df = data_store.get(key)
                    data_dfs.append(df)

            data = pd.concat(data_dfs)
            data.sort_values(
                by=[col for col in ["time_step", "point_idx"] if col in data.columns], inplace=True
            )
        else:
            raise Exception(f"Extension {hist_series_format} not supported for hist series!")
        points = pd.read_csv(points_path)
        return InflowData(data=data, points=points)
