from __future__ import annotations

import pathlib

from pydantic import BaseModel, Field

from cfdmod.use_cases.pressure.statistics import Statistics
from cfdmod.use_cases.pressure.zoning.body_config import BodyConfig
from cfdmod.use_cases.pressure.zoning.processing import MomentVariables
from cfdmod.utils import read_yaml

__all__ = ["CmConfig"]


class CmConfig(BaseModel):
    bodies: dict[str, BodyConfig] = Field(
        ..., title="Bodies definition", description="Named bodies definition"
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

    @classmethod
    def from_file(cls, filename: pathlib.Path) -> dict[str, CmConfig]:
        config_dict: dict[str, CmConfig] = {}
        yaml_vals = read_yaml(filename)

        if "bodies" not in yaml_vals.keys():
            raise Exception("There is no body defined in the configuration file")

        for measurement_lbl in yaml_vals["moment_coefficient"].keys():
            body_list: list[str] = yaml_vals["moment_coefficient"][measurement_lbl]["bodies"]
            yaml_vals["moment_coefficient"][measurement_lbl]["bodies"] = {}
            for body_label in body_list:
                if body_label not in yaml_vals["bodies"].keys():
                    raise Exception("Body is not defined in the configuration file")

                body_cfg = BodyConfig.model_validate(yaml_vals["bodies"][body_label])
                yaml_vals["moment_coefficient"][measurement_lbl]["bodies"][
                    body_label
                ] = body_cfg.model_dump()

            cfg = cls.model_validate(yaml_vals["moment_coefficient"][measurement_lbl])
            config_dict[measurement_lbl] = cfg

        return config_dict
