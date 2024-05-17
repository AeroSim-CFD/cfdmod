from __future__ import annotations

__all__ = ["CpConfig", "CpCaseConfig"]

from typing import Literal

from pydantic import Field

from cfdmod.api.configs.hashable import HashableConfig
from cfdmod.use_cases.pressure.base_config import BasePressureCaseConfig, BasePressureConfig


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


class CpCaseConfig(BasePressureCaseConfig):
    pressure_coefficient: dict[str, CpConfig] = Field(
        ...,
        title="Pressure Coefficient configs",
        description="Dictionary with Pressure Coefficient configuration",
    )
