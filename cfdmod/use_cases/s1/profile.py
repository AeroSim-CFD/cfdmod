from __future__ import annotations

__all__ = [
    "Profile",
    "EUCat",
    "NBRCat",
    "z0_cats_EU",
    "p_cats_NBR",
    "b_cats_NBR",
    "get_EU_u_profile",
    "get_NBR_u_profile",
    "get_EU_cat_u_profile",
    "get_NBR_cat_u_profile",
]

import pathlib
from typing import Literal

import numpy as np
import pandas as pd

EUCat = Literal["0", "I", "II", "III", "IV"]
NBRCat = Literal["I", "II", "III", "IV", "V"]

# EU Params
z0_cats_EU = {"0": 0.003, "I": 0.01, "II": 0.05, "III": 0.3, "IV": 1}
# NBR Params
p_cats_NBR = {"I": 0.095, "II": 0.15, "III": 0.185, "IV": 0.23, "V": 0.31}
b_cats_NBR = {"I": 1.23, "II": 1.00, "III": 0.86, "IV": 0.71, "V": 0.50}


def get_EU_u_profile(
    *, z: np.ndarray, H: float, z0: float, u_ref: float = 1, Fr: float = 0.65
) -> np.ndarray:
    arr_eu = np.log(z / z0) / np.log(H / z0)
    return arr_eu * u_ref


def get_NBR_u_profile(
    *, z: np.ndarray, H: float, b: float, p: float, u_ref: float = 1, Fr: float = 0.65
) -> np.ndarray:
    S2 = lambda z: Fr * b * (z / 10) ** p
    arr_nbr = S2(z) / S2(H)
    return arr_nbr * u_ref


def get_EU_cat_u_profile(
    *, z: np.ndarray, H: float, cat: EUCat, u_ref: float = 1, Fr: float = 0.65
) -> np.ndarray:
    return get_EU_u_profile(z=z, H=H, z0=z0_cats_EU[cat], u_ref=u_ref, Fr=Fr)


def get_NBR_cat_u_profile(
    *, z: np.ndarray, H: float, cat: NBRCat, u_ref: float = 1, Fr: float = 0.65
) -> np.ndarray:
    return get_NBR_u_profile(z=z, H=H, p=p_cats_NBR[cat], b=b_cats_NBR[cat], u_ref=u_ref, Fr=Fr)


def get_EU_Iu_profile(*, z: np.ndarray, z0: float) -> np.ndarray:
    return 1 / np.log(z / z0)


def get_EU_cat_Iu_profile(*, z: np.ndarray, cat: EUCat) -> np.ndarray:
    return get_EU_Iu_profile(z=z, z0=z0_cats_EU[cat])


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
