from __future__ import annotations

import copy
import pathlib

import numpy as np
import pandas as pd


class Profile:
    def __init__(self, pos: np.ndarray, values: np.ndarray, label: str):
        self.pos = pos
        self.values = values
        self.label = label

    def __repr__(self):
        return f"pos: {self.pos} \n values: {self.values}"

    def __truediv__(self, rhs: Profile) -> Profile:
        self_copy = copy.copy(self)
        rhs_copy = copy.copy(rhs)

        self_copy.normalize_position()
        rhs_copy.normalize_position()

        max_height = min(self_copy.pos.max(), rhs_copy.pos.max())
        self_copy.truncate_position(max_height)
        rhs_copy.truncate_position(max_height)

        if max_height not in self_copy.pos:
            self_copy.interpolate_value(max_height)
        elif max_height not in rhs_copy.pos:
            rhs_copy.interpolate_value(max_height)

        [self_copy.interpolate_value(val) for val in np.setdiff1d(rhs_copy.pos, self_copy.pos)]
        [rhs_copy.interpolate_value(val) for val in np.setdiff1d(self_copy.pos, rhs_copy.pos)]

        s1_values = self_copy.values[1:] / rhs_copy.values[1:]  # Ignore wall values (u=0)
        s1_pos = self_copy.pos[1:]  # Ignore wall values (u=0)

        return Profile(s1_pos, s1_values, f"S1: {self_copy.label} / {rhs_copy.label}")

    def normalize_position(self):
        """Normalizes the profile position"""

        min_pos = self.pos.min()
        self.pos -= min_pos

    def truncate_position(self, max_height: float):
        """Truncate the profile given a maximum height"""

        slice_index = np.searchsorted(self.pos, max_height, side="right")
        self.pos = self.pos[:slice_index]
        self.values = self.values[:slice_index]

    def interpolate_value(self, new_pos: float):
        """Interpolate the value given a new position"""

        if new_pos not in self.pos:
            insert_idx = np.searchsorted(self.pos, new_pos)
            interp_val = np.interp(new_pos, self.pos, self.values)

            self.pos = np.insert(self.pos, insert_idx, new_pos)
            self.values = np.insert(self.values, insert_idx, interp_val)

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
