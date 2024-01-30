from __future__ import annotations

__all__ = ["CmConfig", "CmCaseConfig"]

import pathlib
from typing import Optional

from pydantic import BaseModel, Field, model_validator

from cfdmod.api.configs.hashable import HashableConfig
from cfdmod.api.geometry.transformation_config import TransformationConfig
from cfdmod.use_cases.pressure.extreme_values import ExtremeValuesParameters
from cfdmod.use_cases.pressure.statistics import Statistics
from cfdmod.use_cases.pressure.zoning.body_config import BodyConfig, BodyDefinition
from cfdmod.use_cases.pressure.zoning.processing import MomentVariables
from cfdmod.use_cases.pressure.zoning.zoning_model import ZoningModel
from cfdmod.utils import read_yaml


class CmConfig(HashableConfig):
    bodies: list[BodyConfig] = Field(
        ...,
        title="Bodies configuration",
        description="Define which bodies should be processed separated and then joined"
        + "and assign to each a zoning config",
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
    transformation: TransformationConfig = Field(
        ...,
        title="Transformation config",
        description="Configuration for mesh transformation",
    )


class CmCaseConfig(BaseModel):
    bodies: dict[str, BodyDefinition] = Field(
        ..., title="Bodies definition", description="Named bodies definition"
    )
    moment_coefficient: dict[str, CmConfig] = Field(
        ...,
        title="Moment Coefficient configs",
        description="Dictionary with Moment Coefficient configuration",
    )
    extreme_values: Optional[ExtremeValuesParameters] = Field(
        None,
        title="Extreme values parameter",
        description="Parameters for performing extreme value analysis",
    )

    @model_validator(mode="after")
    def check_extreme_values_params(self) -> CmCaseConfig:
        full_stats = [s for v in self.moment_coefficient.values() for s in v.statistics]
        if any(stats in full_stats for stats in ["xtr_min", "xtr_max"]):
            if self.extreme_values is None:
                raise ValueError("Extreme values parameters must be specified!")
        return self

    @model_validator(mode="after")
    def valdate_body_list(self):
        for body_label in [cfg.body for cfg in self.moment_coefficient.values()]:
            if body_label not in self.bodies.keys():
                raise Exception(f"Body {body_label} is not defined in the configuration file")
        return self

    @classmethod
    def from_file(cls, filename: pathlib.Path) -> CmCaseConfig:
        yaml_vals = read_yaml(filename)
        cfg = cls(**yaml_vals)
        return cfg
