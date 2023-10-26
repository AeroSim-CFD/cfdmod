from __future__ import annotations

import pathlib

from pydantic import BaseModel, Field, model_validator

from cfdmod.use_cases.pressure.statistics import Statistics
from cfdmod.use_cases.pressure.zoning.body_config import BodyConfig
from cfdmod.use_cases.pressure.zoning.processing import MomentVariables
from cfdmod.utils import read_yaml

__all__ = ["CmConfig", "CmCaseConfig"]


class CmCaseConfig(BaseModel):
    bodies: dict[str, BodyConfig] = Field(
        ..., title="Bodies definition", description="Named bodies definition"
    )
    moment_coefficient: dict[str, CmConfig] = Field(
        ...,
        title="Moment Coefficient configs",
        description="Dictionary with Moment Coefficient configuration",
    )

    @model_validator(mode="after")
    def valdate_body_list(self):
        for body_label in [b for cfg in self.moment_coefficient.values() for b in cfg.bodies]:
            if body_label not in self.bodies.keys():
                raise Exception(f"Body {body_label} is not defined in the configuration file")
        return self

    @classmethod
    def from_file(cls, filename: pathlib.Path) -> CmCaseConfig:
        yaml_vals = read_yaml(filename)
        cfg = cls(**yaml_vals)
        return cfg


class CmConfig(BaseModel):
    bodies: list[str] = Field(
        ..., title="Bodies definition", description="List of bodies to be processed"
    )
    variables: list[MomentVariables]
    lever_origin: tuple[float, float, float] = Field(
        ...,
        title="Lever origin",
        description="Coordinate of the reference point to evaluate the lever for moment calculations",
    )
    statistics: list[Statistics] = Field(
        ...,
        title="List of statistics",
        description="Define which statistical analysis will be performed to the coefficient",
    )
