from __future__ import annotations

__all__ = ["CpConfig"]

import pathlib
from typing import Literal

from pydantic import BaseModel, Field

from cfdmod.use_cases.pressure.statistics import Statistics
from cfdmod.utils import read_yaml


class CpConfig(BaseModel):
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

    @classmethod
    def from_file(cls, filename: pathlib.Path) -> CpConfig:
        yaml_vals = read_yaml(filename)
        cfg = cls(**yaml_vals["pressure_coefficients"])
        return cfg
