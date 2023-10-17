from __future__ import annotations

__all__ = ["ZoningConfig", "ZoningModel"]

import itertools
import pathlib
from typing import Optional

import pandas as pd
from pydantic import BaseModel, Field, validator

from cfdmod.utils import read_yaml


class ZoningConfig(BaseModel):
    global_zoning: ZoningModel = Field(
        ...,
        title="Global Zoning Config",
        description="Default Zoning Config applied to all surfaces",
    )
    no_zoning: list[str] = Field(
        [],
        title="No Zoning Surfaces",
        description="List of surfaces to ignore region mesh generation",
    )
    exclude: list[str] = Field(
        [],
        title="Surfaces to exclude",
        description="List of surfaces to ignore when calculating shape coefficient",
    )
    exceptions: dict[str, ExceptionZoningModel] = Field(
        {},
        title="Dict with specific zoning",
        description="Define specific zoning config to specific surfaces."
        + "It overrides the global zoning config.",
    )

    @validator("exceptions", always=True)
    def validate_exceptions(cls, v):
        exceptions = []
        for exception_cfg in v.values():
            exceptions += exception_cfg.surfaces
        if len(exceptions) != len(set(exceptions)):
            raise Exception("Invalid exceptions list, surface names must not repeat")
        return v

    @validator("no_zoning", "exclude", always=True)
    def validate_surface_list(cls, v):
        if len(v) != len(set(v)):
            raise Exception("Invalid surface list, names must not repeat")
        return v

    @property
    def surfaces_in_exception(self) -> list[str]:
        exceptions = []
        for exception_cfg in self.exceptions.values():
            exceptions += exception_cfg.surfaces
        return exceptions

    @classmethod
    def from_file(cls, filename: pathlib.Path) -> ZoningConfig:
        yaml_vals = read_yaml(filename)
        cfg = cls.model_validate(yaml_vals)
        return cfg


class ZoningModel(BaseModel):
    x_intervals: list[float] = Field(
        ...,
        title="X intervals list",
        description="Values for the X axis intervals list, it must be unique",
    )
    y_intervals: list[float] = Field(
        ...,
        title="Y intervals list",
        description="Values for the Y axis intervals list, it must be unique",
    )
    z_intervals: list[float] = Field(
        ...,
        title="Z intervals list",
        description="Values for the Z axis intervals list, it must be unique",
    )

    @validator("x_intervals", "y_intervals", "z_intervals", always=True)
    def validate_interval(cls, v):
        if len(v) != len(set(v)):
            raise Exception("Invalid region intervals, values must not repeat")
        return v

    @validator("x_intervals", "y_intervals", "z_intervals", always=True)
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
            new_zoning.x_intervals = [new_zoning.x_intervals[0], new_zoning.x_intervals[-1]]
        elif axis == 1:
            new_zoning.y_intervals = [new_zoning.y_intervals[0], new_zoning.y_intervals[-1]]
        elif axis == 2:
            new_zoning.z_intervals = [new_zoning.z_intervals[0], new_zoning.z_intervals[-1]]

        return new_zoning

    def offset_limits(self, offset_value: float) -> ZoningModel:
        """Add a new offset to the intervals limits to account for mesh deformations

        Args:
            offset_value (float): Offset value to add or subtract from the limits
        """
        offsetted_zoning = self.model_copy()
        offsetted_zoning.x_intervals[0] -= offset_value
        offsetted_zoning.x_intervals[-1] += offset_value
        offsetted_zoning.y_intervals[0] -= offset_value
        offsetted_zoning.y_intervals[-1] += offset_value
        offsetted_zoning.z_intervals[0] -= offset_value
        offsetted_zoning.z_intervals[-1] += offset_value

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
                ["x_min", "x_max", "y_min", "y_max", "z_min", "z_max", "region_index"]
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
        df_regions["region_index"] = df_regions.index

        return df_regions


class ExceptionZoningModel(ZoningModel):
    surfaces: list[str] = Field(
        ...,
        title="List of surfaces",
        description="List of surfaces to include in the exceptional zoning",
    )

    @validator("surfaces", always=True)
    def validate_surface_list(cls, v):
        if len(v) != len(set(v)):
            raise Exception("Invalid exceptions surface list, names must not repeat")
        if len(v) == 0:
            raise Exception("Invalid exceptions surface list, list must not be empty")
        return v
