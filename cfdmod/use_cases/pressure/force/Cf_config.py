from __future__ import annotations

import pathlib

from pydantic import BaseModel, Field, model_validator

from cfdmod.use_cases.pressure.statistics import Statistics
from cfdmod.use_cases.pressure.zoning.body_config import BodyConfig
from cfdmod.use_cases.pressure.zoning.processing import ForceVariables
from cfdmod.utils import read_yaml

__all__ = ["CfConfig", "CfCaseConfig"]


class CfCaseConfig(BaseModel):
    bodies: dict[str, BodyConfig] = Field(
        ..., title="Bodies definition", description="Named bodies definition"
    )
    force_coefficient: dict[str, CfConfig] = Field(
        ...,
        title="Force Coefficient configs",
        description="Dictionary with Force Coefficient configuration",
    )

    @model_validator(mode="after")
    def valdate_body_list(self):
        for body_label in [b for cfg in self.force_coefficient.values() for b in cfg.bodies]:
            if body_label not in self.bodies.keys():
                raise Exception(f"Body {body_label} is not defined in the configuration file")
        return self

    @classmethod
    def from_file(cls, filename: pathlib.Path) -> CfCaseConfig:
        yaml_vals = read_yaml(filename)
        cfg = cls(**yaml_vals)
        return cfg


class CfConfig(BaseModel):
    bodies: list[str] = Field(
        ..., title="Bodies definition", description="List of bodies to be processed"
    )
    variables: list[ForceVariables] = Field(
        ...,
        title="List of variables",
        description="Define which variables will be calculated",
    )
    statistics: list[Statistics] = Field(
        ...,
        title="List of statistics",
        description="Define which statistical analysis will be performed to the coefficient",
    )
