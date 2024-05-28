from __future__ import annotations

__all__ = ["CpConfig", "CpCaseConfig"]

import pathlib
from typing import Literal

from pydantic import BaseModel, Field

from cfdmod.api.configs.hashable import HashableConfig
from cfdmod.use_cases.pressure.base_config import BasePressureConfig
from cfdmod.utils import read_yaml


class CpConfig(HashableConfig, BasePressureConfig):
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
    simul_U_H: float = Field(
        ...,
        title="Simulation Flow Velocity",
        description="Value for simulation Flow Velocity to calculate dynamic "
        + "pressure and convert time scales",
    )
    simul_characteristic_length: float = Field(
        ...,
        title="Simulation Characteristic Length",
        description="Value for simulation characteristic length to convert time scales",
    )


class CpCaseConfig(BaseModel):
    pressure_coefficient: dict[str, CpConfig] = Field(
        ...,
        title="Pressure Coefficient configs",
        description="Dictionary with Pressure Coefficient configuration",
    )

    @classmethod
    def from_file(cls, filename: pathlib.Path) -> CpCaseConfig:
        yaml_vals = read_yaml(filename)
        cfg = cls(**yaml_vals)
        return cfg
