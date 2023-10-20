from __future__ import annotations

__all__ = ["ZoningConfig"]

import pathlib

from pydantic import BaseModel, Field, validator

from cfdmod.utils import read_yaml
from cfdmod.use_cases.pressure.zoning.zoning_model import ZoningModel

class ZoningConfig(BaseModel):
    global_zoning: ZoningModel = Field(
        ...,
        title="Global Zoning Config",
        description="Default Zoning Config applied to all surfaces",
    )
    no_zoning: list[str] = Field(
        [],
        title="No Zoning Surfaces",
        description="List of surfaces to ignore region mesh generation",
    )
    exclude: list[str] = Field(
        [],
        title="Surfaces to exclude",
        description="List of surfaces to ignore when calculating shape coefficient",
    )
    exceptions: dict[str, ExceptionZoningModel] = Field(
        {},
        title="Dict with specific zoning",
        description="Define specific zoning config to specific surfaces."
        + "It overrides the global zoning config.",
    )

    @validator("exceptions", always=True)
    def validate_exceptions(cls, v):
        exceptions = []
        for exception_cfg in v.values():
            exceptions += exception_cfg.surfaces
        if len(exceptions) != len(set(exceptions)):
            raise Exception("Invalid exceptions list, surface names must not repeat")
        return v

    @validator("no_zoning", "exclude", always=True)
    def validate_surface_list(cls, v):
        if len(v) != len(set(v)):
            raise Exception("Invalid surface list, names must not repeat")
        return v

    @property
    def surfaces_in_exception(self) -> list[str]:
        exceptions = []
        for exception_cfg in self.exceptions.values():
            exceptions += exception_cfg.surfaces
        return exceptions

    @classmethod
    def from_file(cls, filename: pathlib.Path) -> ZoningConfig:
        yaml_vals = read_yaml(filename)
        cfg = cls.model_validate(yaml_vals)
        return cfg


class ExceptionZoningModel(ZoningModel):
    surfaces: list[str] = Field(
        ...,
        title="List of surfaces",
        description="List of surfaces to include in the exceptional zoning",
    )

    @validator("surfaces", always=True)
    def validate_surface_list(cls, v):
        if len(v) != len(set(v)):
            raise Exception("Invalid exceptions surface list, names must not repeat")
        if len(v) == 0:
            raise Exception("Invalid exceptions surface list, list must not be empty")
        return v
