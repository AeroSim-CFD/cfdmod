from __future__ import annotations

import pathlib

from pydantic import BaseModel, Field, validator

from cfdmod.use_cases.pressure.shape.zoning_config import ZoningConfig
from cfdmod.use_cases.pressure.statistics import Statistics
from cfdmod.utils import read_yaml

__all__ = ["CeConfig", "CeCaseConfig"]


class ZoningBuilder(BaseModel):
    yaml: str = Field(
        ...,
        title="Path to Zoning yaml",
        description="Path to Zoning yaml for construction zoning configuration",
    )

    def to_zoning_config(self) -> ZoningConfig:
        zoning_cfg = ZoningConfig.from_file(pathlib.Path(self.yaml))
        return zoning_cfg


class CeConfig(BaseModel):
    """Configuration for shape coefficient"""

    zoning: ZoningConfig | ZoningBuilder = Field(
        ...,
        title="Zoning configuration",
        description="Zoning configuration with intervals information",
    )
    statistics: list[Statistics] = Field(
        ...,
        title="List of statistics",
        description="List of statistics to calculate from shape coefficient signal",
    )

    @validator("zoning", always=True)
    def validate_zoning(cls, v):
        if isinstance(v, ZoningBuilder):
            return v.to_zoning_config()
        else:
            return v


class CeCaseConfig(BaseModel):
    shape_coefficient: dict[str, CeConfig] = Field(
        ...,
        title="Shape Coefficient configs",
        description="Dictionary of shape coefficient configurations",
    )

    @classmethod
    def from_file(cls, filename: pathlib.Path) -> CeCaseConfig:
        yaml_vals = read_yaml(filename)
        cfg = cls(**yaml_vals)
        return cfg
