from __future__ import annotations

import pathlib
from dataclasses import dataclass
from typing import Literal

import pandas as pd

from cfdmod.analysis.inflow.functions import spectral_density

VelocityComponents = Literal["ux", "uy", "uz"]


@dataclass
class NormalizationParameters:
    reference_velocity: float
    characteristic_length: float


class InflowData:
    def __init__(self, data: pd.DataFrame, points: pd.DataFrame):
        self.data = data
        self.points = points

    def calculate_mean_velocity(self, for_components: list[VelocityComponents]) -> pd.DataFrame:
        """Calculates the turbulence intensity for each component given

        Args:
            for_components (list[VelocityComponents]): List of components to calculate mean velocity from

        Returns:
            pd.DataFrame: Mean Velocity dataframe
        """
        if not all(
            [component in self.data.columns for component in for_components + ["point_idx"]]
        ):
            raise ValueError("Components must be inside inflow profile data columns")

        group_by_point_idx = self.data.groupby("point_idx")
        velocity_data = group_by_point_idx.agg(
            {component: "mean" for component in for_components}
        ).reset_index()
        # Rename columns
        velocity_data.columns = [
            col + "_mean" if col in for_components else col for col in velocity_data.columns
        ]
        return velocity_data

    def calculate_turbulence_intensity(
        self, for_components: list[VelocityComponents]
    ) -> pd.DataFrame:
        """Calculates the turbulence intensity for each component given

        Args:
            for_components (list[VelocityComponents]): List of components to calculate turbulence intensity from

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
        # Rename columns
        turbulence_data.columns = [
            "_".join(col) if col[1] != "" else col[0] for col in turbulence_data.columns
        ]
        for component in for_components:
            turbulence_data[f"I_{component}"] = (
                turbulence_data[f"{component}_std"] / turbulence_data[f"{component}_mean"]
            )

        return turbulence_data[["point_idx"] + [f"I_{component}" for component in for_components]]

    def calculate_spectral_density(
        self,
        target_index: int,
        for_components: list[VelocityComponents],
        normalization_params: NormalizationParameters,
    ) -> pd.DataFrame:
        """Calculates the spectral density for a given target point index

        Args:
            target_index (int): Index of the target point
            for_components (list[VelocityComponents]): List of components to calculate turbulence intensity from
            normalization_params (NormalizationParameters): Parameters for spectral density normalization

        Returns:
            pd.DataFrame: Spectral density data
        """
        spectral_data = pd.DataFrame()
        for component in for_components:
            point_data = self.data.loc[self.data["point_idx"] == target_index]
            vel_arr = point_data[component].to_numpy()
            time_arr = point_data["time_step"].to_numpy()

            spec_dens, norm_freq = spectral_density(
                velocity_signal=vel_arr,
                timestamps=time_arr,
                reference_velocity=normalization_params.reference_velocity,
                characteristic_length=normalization_params.characteristic_length,
            )
            spectral_data[f"S ({component})"] = spec_dens
            spectral_data[f"f ({component})"] = norm_freq

        return spectral_data

    def calculate_autocorrelation(
        self, anchor_point_idx: int, for_components: list[VelocityComponents]
    ) -> pd.DataFrame:
        """Calculates the autocorrelation from an anchor point

        Args:
            anchor_point_idx (int): Index of the anchor point
            for_components (list[VelocityComponents]): List of components to calculate turbulence intensity from

        Returns:
            pd.DataFrame: Autocorrelation data
        """
        anchor_data = self.data.loc[self.data["point_idx"] == anchor_point_idx].copy()
        anchor_data = anchor_data[for_components + ["point_idx", "time_step"]]
        for component in for_components:
            anchor_data[f"{component}_a"] = anchor_data[f"{component}"]
            anchor_data[f"{component}_a^2"] = anchor_data[f"{component}"] ** 2
        data_to_merge = anchor_data[
            ["time_step"]
            + [f"{component}_a{symbol}" for component in for_components for symbol in ["^2", ""]]
        ]
        merged_data = pd.merge(self.data, data_to_merge, on="time_step", how="left")
        for component in for_components:
            merged_data[f"{component}_{component}_a"] = (
                merged_data[f"{component}"] * merged_data[f"{component}_a"]
            )
        avg_data = merged_data.groupby("point_idx").mean()
        for component in for_components:
            avg_data[f"coef_{component}"] = (
                avg_data[f"{component}_{component}_a"]
                - avg_data[f"{component}"] * avg_data[f"{component}_a"]
            ) / (avg_data[f"{component}_a^2"] - avg_data[f"{component}_a"] ** 2)
        autocorrelation = avg_data[[f"coef_{c}" for c in for_components]].reset_index()
        return autocorrelation

    @classmethod
    def from_files(cls, hist_series_path: pathlib.Path, points_path: pathlib.Path) -> InflowData:
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
