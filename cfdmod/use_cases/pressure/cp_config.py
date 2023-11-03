from __future__ import annotations

__all__ = ["CpConfig"]

import pathlib
from typing import Literal

from pydantic import BaseModel, Field

from cfdmod.api.configs.hashable import HashableConfig
from cfdmod.use_cases.pressure.statistics import Statistics
from cfdmod.utils import read_yaml

__all__ = ["CpConfig", "CpCaseConfig"]


class CpConfig(HashableConfig):
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
    statistics: list[Statistics] = Field(
        ...,
        title="List of statistics",
        description="List of statistics to calculate from pressure coefficient signal",
    )


class CpCaseConfig(BaseModel):
    pressure_coefficient: CpConfig = Field(
        ...,
        title="Cp configuration",
        description="Configuration with pressure coefficient post processing configs",
    )

    @classmethod
    def from_file(cls, filename: pathlib.Path) -> CpCaseConfig:
        yaml_vals = read_yaml(filename)
        cfg = cls(**yaml_vals)
        return cfg
