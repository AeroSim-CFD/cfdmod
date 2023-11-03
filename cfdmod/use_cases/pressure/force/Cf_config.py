from __future__ import annotations

import pathlib

from pydantic import BaseModel, Field, model_validator

from cfdmod.use_cases.pressure.statistics import Statistics
from cfdmod.use_cases.pressure.zoning.body_config import BodyConfig
from cfdmod.use_cases.pressure.zoning.processing import ForceVariables
from cfdmod.use_cases.pressure.zoning.zoning_model import ZoningModel
from cfdmod.utils import read_yaml

__all__ = ["CfConfig", "CfCaseConfig"]


class CfConfig(BaseModel):
    body: str = Field(..., title="Body label", description="Define which body should be processed")
    sub_bodies: ZoningModel = Field(
        ZoningModel(
            x_intervals=[float("-inf"), float("inf")],
            y_intervals=[float("-inf"), float("inf")],
            z_intervals=[float("-inf"), float("inf")],
        ),
        title="Sub body intervals",
        description="Definition of the intervals that will section the body into sub-bodies",
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
        for body_label in [cfg.body for cfg in self.force_coefficient.values()]:
            if body_label not in self.bodies.keys():
                raise Exception(f"Body {body_label} is not defined in the configuration file")
        return self

    @classmethod
    def from_file(cls, filename: pathlib.Path) -> CfCaseConfig:
        yaml_vals = read_yaml(filename)
        cfg = cls(**yaml_vals)
        return cfg
