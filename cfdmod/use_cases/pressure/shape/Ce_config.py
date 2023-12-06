from __future__ import annotations

import pathlib

from pydantic import BaseModel, Field, field_validator

from cfdmod.api.configs.hashable import HashableConfig
from cfdmod.api.geometry.transformation_config import TransformationConfig
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
    _base_path: pathlib.Path = pathlib.Path("./")

    def to_zoning_config(self) -> ZoningConfig:
        zoning_cfg = ZoningConfig.from_file(self._base_path / pathlib.Path(self.yaml))
        return zoning_cfg


class CeConfig(HashableConfig):
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
    sets: dict[str, list[str]] = Field(
        {}, title="Surface sets", description="Combine multiple surfaces into a set of surfaces"
    )
    transformation: TransformationConfig = Field(
        ...,
        title="Transformation config",
        description="Configuration for mesh transformation",
    )

    @property
    def surfaces_in_sets(self):
        surface_list = [sfc for sfc_list in self.sets.values() for sfc in sfc_list]
        return surface_list

    @field_validator("sets")
    def validate_sets(cls, v):
        surface_list = [sfc for sfc_list in v.values() for sfc in sfc_list]
        if len(surface_list) != len(set(surface_list)):
            raise Exception(f"A surface cannot be listed in more than one set")
        return v

    def to_zoning(self):
        if isinstance(self.zoning, ZoningBuilder):
            self.zoning = self.zoning.to_zoning_config()

    def validate_zoning_surfaces(self):
        common_surfaces = set(self.surfaces_in_sets).intersection(set(self.zoning.surfaces_listed))  # type: ignore
        if len(common_surfaces) != 0:
            raise Exception("Surfaces inside a set cannot be listed in zoning")


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
        for s in cfg.shape_coefficient.values():
            s.zoning._base_path = filename.parent
            s.to_zoning()
            s.validate_zoning_surfaces()

        return cfg
