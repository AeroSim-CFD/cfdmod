from __future__ import annotations

import pathlib

from pydantic import BaseModel, Field

from cfdmod.use_cases.pressure.force.body_config import BodyConfig
from cfdmod.use_cases.pressure.statistics import Statistics
from cfdmod.use_cases.pressure.zoning.processing import ForceVariables
from cfdmod.utils import read_yaml

__all__ = ["CfConfig"]


class CfConfig(BaseModel):
    bodies: dict[str, BodyConfig] = Field(
        ..., title="Bodies definition", description="Named bodies definition"
    )
    variables: list[ForceVariables]
    statistics: list[Statistics]

    @classmethod
    def from_file(cls, filename: pathlib.Path) -> dict[str, CfConfig]:
        config_dict: dict[str, CfConfig] = {}
        yaml_vals = read_yaml(filename)

        if "bodies" not in yaml_vals.keys():
            raise Exception("There is no body defined in the configuration file")

        for measurement_lbl in yaml_vals["force_coefficient"].keys():
            body_list: list[str] = yaml_vals["force_coefficient"][measurement_lbl]["bodies"]
            yaml_vals["force_coefficient"][measurement_lbl]["bodies"] = {}
            for body_label in body_list:
                if body_label not in yaml_vals["bodies"].keys():
                    raise Exception("Body is not defined in the configuration file")

                body_cfg = BodyConfig.model_validate(yaml_vals["bodies"][body_label])
                yaml_vals["force_coefficient"][measurement_lbl]["bodies"][
                    body_label
                ] = body_cfg.model_dump()

            cfg = cls.model_validate(yaml_vals["force_coefficient"][measurement_lbl])
            config_dict[measurement_lbl] = cfg

        return config_dict
