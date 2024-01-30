from __future__ import annotations

__all__ = ["CpConfig", "CpCaseConfig"]

import pathlib
from typing import Literal, Optional

from pydantic import BaseModel, Field, model_validator

from cfdmod.api.configs.hashable import HashableConfig
from cfdmod.use_cases.pressure.extreme_values import ExtremeValuesParameters
from cfdmod.use_cases.pressure.statistics import Statistics
from cfdmod.utils import read_yaml


class CpConfig(HashableConfig):
    number_of_chunks: int = Field(
        1,
        title="Number of chunks",
        description="How many chunks the output time series will be split into",
        ge=1,
    )
    timestep_range: tuple[float, float] = Field(
        ...,
        title="Timestep Range",
        description="Interval between start and end steps to slice data",
    )
    reference_pressure: Literal["average", "instantaneous"] = Field(
        ...,
        title="Reference Pressure",
        description="Sets how to account for reference pressure effects."
        + "If set to average, static pressure signal will be averaged."
        + "If set to instantaneous, static pressure signal will be transient.",
    )
    U_H: float = Field(
        ...,
        title="Reference Flow Velocity",
        description="Value for reference Flow Velocity to calculate dynamic pressure",
    )
    U_H_correction_factor: float = Field(
        1,
        title="Reference Flow Velocity correction factor",
        description="Value for reference Flow Velocity correction factor multiplier",
    )
    statistics: list[Statistics] = Field(
        ...,
        title="List of statistics",
        description="List of statistics to calculate from pressure coefficient signal",
    )


class CpCaseConfig(BaseModel):
    pressure_coefficient: dict[str, CpConfig] = Field(
        ...,
        title="Pressure Coefficient configs",
        description="Dictionary with Pressure Coefficient configuration",
    )
    extreme_values: Optional[ExtremeValuesParameters] = Field(
        None,
        title="Extreme values parameter",
        description="Parameters for performing extreme value analysis",
    )

    @model_validator(mode="after")
    def check_extreme_values_params(self) -> CpCaseConfig:
        full_stats = [s for v in self.pressure_coefficient.values() for s in v.statistics]
        if any(stats in full_stats for stats in ["xtr_min", "xtr_max"]):
            if self.extreme_values is None:
                raise ValueError("Extreme values parameters must be specified!")
        return self

    @classmethod
    def from_file(cls, filename: pathlib.Path) -> CpCaseConfig:
        yaml_vals = read_yaml(filename)
        cfg = cls(**yaml_vals)
        return cfg
