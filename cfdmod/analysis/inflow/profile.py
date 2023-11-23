from __future__ import annotations

import pathlib
from typing import Literal

import numpy as np
import pandas as pd

VelocityComponents = Literal["ux", "uy", "uz"]


class InflowProfile:
    def __init__(self, data: pd.DataFrame, points: pd.DataFrame):
        self.data = data
        self.points = points

    def calculate_turbulence_intensity(
        self, for_components: list[VelocityComponents]
    ) -> pd.DataFrame:
        """Calculates the turbulence intensity for each component given

        Args:
            for_components (list[VelocityComponents]): List of components to calculate turbulence intensity from

        Raises:
            ValueError: If the data does not contain a column for one of the components

        Returns:
            pd.DataFrame: Turbulence intensity dataframe
        """
        if not all(
            [component in self.data.columns for component in for_components + ["point_idx"]]
        ):
            raise ValueError("Components must be inside inflow profile data columns")

        group_by_point_idx = self.data.groupby("point_idx")
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

        return turbulence_data[["point_idx"] + [f"I_{component}" for component in for_components]]

    @classmethod
    def from_folder(cls, folder_path: pathlib.Path) -> InflowProfile:
        data = pd.read_csv(folder_path / "hist_series.csv")
        points = pd.read_csv(folder_path / "points.csv")
        return InflowProfile(data=data, points=points)
