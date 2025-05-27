from __future__ import annotations

__all__ = ["CpConfig", "CpCaseConfig"]

import pathlib
from typing import Annotated, Literal

from pydantic import BaseModel, Field

from cfdmod.api.configs.hashable import HashableConfig
from cfdmod.use_cases.pressure.base_config import BasePressureConfig
from cfdmod.utils import read_yaml


class CpConfig(HashableConfig, BasePressureConfig):
    """Configuration to calculate pressure coeficient"""

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
    macroscopic_type: Annotated[
        Literal["rho", "pressure"],
        Field(
            "rho",
            title="Macroscopic type",
            description="Macroscopic type in files, LBM density (rho) or pressure",
        ),
    ]
    fluid_density: Annotated[
        float,
        Field(
            1, title="Fluid density", description="Fluid density to consider for Cp calculation"
        ),
    ]
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
