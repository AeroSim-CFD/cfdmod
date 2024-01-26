from __future__ import annotations

__all__ = ["CfConfig", "CfCaseConfig"]

import pathlib
from typing import Optional

from pydantic import BaseModel, Field, model_validator

from cfdmod.api.configs.hashable import HashableConfig
from cfdmod.api.geometry.transformation_config import TransformationConfig
from cfdmod.use_cases.pressure.extreme_values import ExtremeValuesParameters
from cfdmod.use_cases.pressure.statistics import Statistics
from cfdmod.use_cases.pressure.zoning.body_config import BodyConfig
from cfdmod.use_cases.pressure.zoning.processing import ForceVariables
from cfdmod.use_cases.pressure.zoning.zoning_model import ZoningModel
from cfdmod.utils import read_yaml


class CfConfig(HashableConfig):
    # body: str = Field(..., title="Body label", description="Define which body should be processed")
    bodies: list[str] = Field(
        ..., title="Body label", description="Define which body should be processed"
    )
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
    transformation: TransformationConfig = Field(
        ...,
        title="Transformation config",
        description="Configuration for mesh transformation",
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
    extreme_values: Optional[ExtremeValuesParameters] = Field(
        None,
        title="Extreme values parameter",
        description="Parameters for performing extreme value analysis",
    )

    @model_validator(mode="after")
    def check_extreme_values_params(self) -> CfCaseConfig:
        full_stats = [s for v in self.force_coefficient.values() for s in v.statistics]
        if any(stats in full_stats for stats in ["xtr_min", "xtr_max"]):
            if self.extreme_values is None:
                raise ValueError("Extreme values parameters must be specified!")
        return self

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
