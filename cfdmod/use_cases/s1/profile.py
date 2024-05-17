from __future__ import annotations

__all__ = ["Profile"]

import pathlib

import numpy as np
import pandas as pd


class Profile:
    def __init__(self, heights: np.ndarray, values: np.ndarray, label: str):
        self.heights = heights
        self.values = values
        self.label = label

    def __repr__(self):
        return f"pos: {self.heights} \n values: {self.values}"

    def update_height_values(self, new_heights: np.ndarray):
        self.values = np.interp(new_heights, self.heights, self.values)
        self.heights = new_heights.copy()

    def copy(self) -> Profile:
        return Profile(heights=self.heights.copy(), values=self.values.copy(), label=self.label)

    def __truediv__(self, rhs: Profile) -> Profile:
        self_copy = self.copy()
        rhs_copy = rhs.copy()

        self_copy.normalize_position()
        rhs_copy.normalize_position()

        max_height = min(self_copy.heights.max(), rhs_copy.heights.max())
        self_copy.truncate_position(max_height)
        rhs_copy.truncate_position(max_height)

        # pos_use = np.append(self_copy.heights, rhs_copy.heights, axis=0)
        pos_use = self_copy.heights.copy()

        rhs_copy.update_height_values(pos_use)

        mask_use = np.abs(rhs_copy.values) > 1e-6
        mask_use[0] = False  # Ignore wall values (u=0)
        s1 = self_copy.values[mask_use] / rhs_copy.values[mask_use]
        s1_heights = self_copy.heights[mask_use]  # Ignore wall values (u=0)

        return Profile(s1_heights, s1, f"S1: {self_copy.label} / {rhs_copy.label}")

    def smoothen_values(self):
        """Removes duplicate values from the profile.
        Duplicate values are a result of probing a vtm with more resolution than the multiblock data.
        """
        dup_indices = np.where(self.values[:-1] == self.values[1:])[0] + 1

        x = self.heights.copy()
        x[dup_indices - 1] = (self.heights[dup_indices] + self.heights[dup_indices - 1]) / 2
        x = np.delete(x, dup_indices)

        y = np.delete(self.values, dup_indices)

        self.heights = x
        self.values = y

    def normalize_position(self):
        """Normalizes the profile position"""

        min_pos = self.heights.min()
        self.heights -= min_pos

    def truncate_position(self, max_height: float):
        """Truncate the profile given a maximum height"""

        slice_index = np.searchsorted(self.heights, max_height, side="right")
        self.heights = self.heights[:slice_index]
        self.values = self.values[:slice_index]

    @classmethod
    def from_csv(
        cls, csv_path: pathlib.Path, position_lbl: str, value_lbl: str, profile_lbl: str
    ) -> Profile:
        """Creates an instance of a Profile from a CSV file

        Args:
            csv_path (pathlib.Path): Path to the CSV file
            position_lbl (str): Label of the column for position values
            value_lbl (str): Label of the column for variable values
            profile_lbl (str): Label of the profile

        Returns:
            Profile: Instance of Profile
        """
        profile_data = pd.read_csv(csv_path)

        if position_lbl not in profile_data.columns:
            raise Exception(f"Data must contain column named {position_lbl}")
        if value_lbl not in profile_data.columns:
            raise Exception(f"Data must contain column named {value_lbl}")

        pos = profile_data[position_lbl].to_numpy()
        values = profile_data[value_lbl].to_numpy()

        return Profile(pos, values, profile_lbl)
