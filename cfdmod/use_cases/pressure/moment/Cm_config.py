from __future__ import annotations

__all__ = ["CmConfig", "CmCaseConfig"]

import pathlib

from pydantic import BaseModel, Field, model_validator

from cfdmod.api.configs.hashable import HashableConfig
from cfdmod.api.geometry.transformation_config import TransformationConfig
from cfdmod.use_cases.pressure.base_config import BasePressureConfig
from cfdmod.use_cases.pressure.zoning.body_config import BodyDefinition, MomentBodyConfig
from cfdmod.use_cases.pressure.zoning.processing import AxisDirections
from cfdmod.utils import read_yaml


class CmConfig(HashableConfig, BasePressureConfig):
    bodies: list[MomentBodyConfig] = Field(
        ...,
        title="Bodies configuration",
        description="Define which bodies should be processed separated and then joined"
        + "and assign to each a zoning config",
    )
    directions: list[AxisDirections] = Field(
        ...,
        title="List of directions",
        description="Define for which directions force coefficient will be calculated",
    )
    transformation: TransformationConfig = Field(
        TransformationConfig(),
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

    @model_validator(mode="after")
    def valdate_body_list(self):
        for body_label in [b.name for cfg in self.moment_coefficient.values() for b in cfg.bodies]:
            if body_label not in self.bodies.keys():
                raise Exception(f"Body {body_label} is not defined in the configuration file")
        return self

    @model_validator(mode="after")
    def valdate_body_surfaces(self):
        for cfg_lbl, cfg in self.moment_coefficient.items():
            all_sfc = [sfc for b in cfg.bodies for sfc in self.bodies[b.name].surfaces]
            if len(all_sfc) != len(set(all_sfc)):
                raise Exception(f"Config {cfg_lbl} repeats surface in more than one body.")
        return self

    @classmethod
    def from_file(cls, filename: pathlib.Path) -> CmCaseConfig:
        yaml_vals = read_yaml(filename)
        cfg = cls(**yaml_vals)
        return cfg
