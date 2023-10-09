from __future__ import annotations

__all__ = ["ZoningConfig"]

import pathlib
from typing import Optional

from pydantic import BaseModel, Field, validator

from cfdmod.utils import read_yaml


class ZoningConfig(BaseModel):
    global_zoning: ZoningModel = Field(
        ...,
        title="Global Zoning Config",
        description="Default Zoning Config applied to all surfaces",
    )
    no_zoning: Optional[list[str]] = Field(
        None,
        title="No Zoning Surfaces",
        description="List of surfaces to ignore region mesh generation",
    )
    exclude: Optional[list[str]] = Field(
        None,
        title="Surfaces to exclude",
        description="List of surfaces to ignore when calculating shape coefficient",
    )
    exceptions: Optional[dict[str, ExceptionZoningModel]] = Field(
        None,
        title="Dict with specific zoning",
        description="Define specific zoning config to specific surfaces."
        + "It overrides the global zoning config.",
    )

    @validator("no_zoning", "exclude", always=True)
    def validate_interval(cls, v):
        if v is None:
            return v
        if len(v) != len(set(v)):
            raise Exception("Invalid surface list, names must not repeat")
        if len(v) == 0:
            raise Exception("Invalid surface list, list must not be empty")
        return v

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


class ExceptionZoningModel(ZoningModel):
    surfaces: list[str] = Field(
        ...,
        title="List of surfaces",
        description="List of surfaces to include in the exceptional zoning",
    )

    @validator("surfaces", always=True)
    def validate_interval(cls, v):
        if len(v) != len(set(v)):
            raise Exception("Invalid surface list, names must not repeat")
        if len(v) == 0:
            raise Exception("Invalid surface list, list must not be empty")
        return v
