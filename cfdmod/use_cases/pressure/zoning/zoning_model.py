from __future__ import annotations

__all__ = ["ZoningModel"]

import itertools

import pandas as pd
from pydantic import BaseModel, Field, field_validator


class ZoningModel(BaseModel):
    x_intervals: list[float] = Field(
        [float("-inf"), float("inf")],
        title="X intervals list",
        description="Values for the X axis intervals list, it must be unique",
    )
    y_intervals: list[float] = Field(
        [float("-inf"), float("inf")],
        title="Y intervals list",
        description="Values for the Y axis intervals list, it must be unique",
    )
    z_intervals: list[float] = Field(
        [float("-inf"), float("inf")],
        title="Z intervals list",
        description="Values for the Z axis intervals list, it must be unique",
    )

    @field_validator("x_intervals", "y_intervals", "z_intervals")
    def validate_interval(cls, v):
        if len(v) != len(set(v)):
            raise Exception("Invalid region intervals, values must not repeat")
        return v

    @field_validator("x_intervals", "y_intervals", "z_intervals")
    def validate_intervals(cls, v):
        if len(v) < 2:
            raise Exception("Interval must have at least 2 values")
        for i in range(len(v) - 1):
            if v[i] >= v[i + 1]:
                raise Exception("Interval must have ascending order")
        return v

    def ignore_axis(self, axis: int) -> ZoningModel:
        """Ignore intervals for a given axis

        Args:
            axis (int): Axis index (x=0, y=1, z=2)
        """
        new_zoning = self.model_copy()
        if axis == 0:
            new_zoning.x_intervals = [float("-inf"), float("inf")]
        elif axis == 1:
            new_zoning.y_intervals = [float("-inf"), float("inf")]
        elif axis == 2:
            new_zoning.z_intervals = [float("-inf"), float("inf")]

        return new_zoning

    def offset_limits(self, offset_value: float) -> ZoningModel:
        """Add a new offset to the intervals limits to account for mesh deformations

        Args:
            offset_value (float): Offset value to add or subtract from the limits
        """
        offsetted_zoning = self.model_copy()

        x_int = self.x_intervals[:]
        y_int = self.y_intervals[:]
        z_int = self.z_intervals[:]

        x_int[0] -= offset_value
        x_int[-1] += offset_value
        y_int[0] -= offset_value
        y_int[-1] += offset_value
        z_int[0] -= offset_value
        z_int[-1] += offset_value

        offsetted_zoning.x_intervals = x_int[:]
        offsetted_zoning.y_intervals = y_int[:]
        offsetted_zoning.z_intervals = z_int[:]

        return offsetted_zoning

    def get_regions(self) -> list[tuple[tuple[float, float], ...]]:
        """Get regions for intervals in each dimension

        Returns:
            list[tuple[tuple[float, float], ...]]: List of regions as
                ((x_min, x_max), (y_min, y_max), (z_min, z_max)) for all intervals combinations
        """
        regions = []

        interval_for_region = lambda intervals: [
            (intervals[i], intervals[i + 1]) for i in range(len(intervals) - 1)
        ]
        x_regions = interval_for_region(self.x_intervals)
        y_regions = interval_for_region(self.y_intervals)
        z_regions = interval_for_region(self.z_intervals)

        regions_iter = itertools.product(x_regions, y_regions, z_regions)
        for region in regions_iter:
            regions.append(region)

        return regions

    def get_regions_df(self) -> pd.DataFrame:
        """Get dataframe for regions of intervals in each dimension

        Returns:
            pd.DataFrame: dataframe of intervals with keys
                ["x_min", "x_max", "y_min", "y_max", "z_min", "z_max", "region_idx"]
        """

        regions = self.get_regions()

        regions_dct = {
            "x_min": [],
            "x_max": [],
            "y_min": [],
            "y_max": [],
            "z_min": [],
            "z_max": [],
        }
        for region in regions:
            for i, d in enumerate(["x", "y", "z"]):
                regions_dct[f"{d}_min"].append(region[i][0])
                regions_dct[f"{d}_max"].append(region[i][1])

        df_regions = pd.DataFrame(regions_dct)
        df_regions["region_idx"] = df_regions.index

        return df_regions
