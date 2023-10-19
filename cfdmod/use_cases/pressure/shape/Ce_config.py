from __future__ import annotations

import pathlib

from pydantic import BaseModel, Field

from cfdmod.use_cases.pressure.shape.zoning_config import ZoningConfig
from cfdmod.use_cases.pressure.statistics import Statistics
from cfdmod.utils import read_yaml

__all__ = ["CeConfig"]


class CeConfig(BaseModel):
    """Configuration for shape coefiecient"""

    zoning: ZoningConfig = Field(
        ...,
        title="Zoning configuration",
        description="Zoning configuration with intervals information",
    )
    statistics: list[Statistics] = Field(
        ...,
        title="List of statistics",
        description="List of statistics to calculate from shape coefficient signal",
    )

    @classmethod
    def from_file(cls, filename: pathlib.Path) -> dict[str, CeConfig]:
        config_dict: dict[str, CeConfig] = {}
        yaml_vals = read_yaml(filename)

        for pattern_lbl in yaml_vals["shape_coefficient"].keys():
            if "yaml" in yaml_vals["shape_coefficient"][pattern_lbl]["zoning"].keys():
                zoning_path = yaml_vals["shape_coefficient"][pattern_lbl]["zoning"]["yaml"]
                zoning_cfg = ZoningConfig.from_file(pathlib.Path(zoning_path))
                yaml_vals["shape_coefficient"][pattern_lbl]["zoning"] = zoning_cfg.model_dump()
                del zoning_path

            cfg = cls.model_validate(yaml_vals["shape_coefficient"][pattern_lbl])
            config_dict[pattern_lbl] = cfg

        return config_dict
