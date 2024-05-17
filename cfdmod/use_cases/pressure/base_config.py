from __future__ import annotations

import pathlib
from typing import Optional

from pydantic import BaseModel, Field, field_validator, model_validator

from cfdmod.use_cases.pressure.extreme_values import TimeScaleParameters
from cfdmod.use_cases.pressure.statistics import BasicStatisticModel, ParameterizedStatisticModel
from cfdmod.utils import read_yaml


class BasePressureConfig(BaseModel):
    statistics: list[BasicStatisticModel | ParameterizedStatisticModel] = Field(
        ...,
        title="List of statistics",
        description="List of statistics to calculate from pressure coefficient signal",
    )

    @field_validator("statistics", mode="before")
    def validate_statistics(cls, v):
        if isinstance(v[0], dict):
            stats_types = [s["stats"] for s in v]
        else:
            stats_types = [s.stats for s in v]
        if len(set(stats_types)) != len(stats_types):
            raise Exception("Duplicated statistics! It can only have one statistic of each type")
        if "mean_eq" in stats_types:
            if any(expected_s not in stats_types for expected_s in ["mean", "min", "max"]):
                raise Exception("Equivalent mean (mean_eq) requires mean, min and max statistics")
        validated_list = []
        for statistic in v:
            if isinstance(statistic, dict):
                if "params" in statistic.keys():
                    validated_list.append(ParameterizedStatisticModel(**statistic))
                else:
                    validated_list.append(BasicStatisticModel(**statistic))
            else:
                validated_list.append(statistic)
        return validated_list


class BasePressureCaseConfig(BaseModel):
    time_scale_conversion: Optional[TimeScaleParameters] = Field(
        None,
        title="Time scale conversion parameters",
        description="Parameters for converting time scale",
    )

    @model_validator(mode="after")
    def check_extreme_values_params(self) -> BasePressureCaseConfig:
        attributes = dir(self)
        attr_lbl = [attr for attr in attributes if attr.endswith("_coefficient")][0]
        case_dict = getattr(self, attr_lbl)
        parameterized_stats = [
            s for v in case_dict.values() for s in v.statistics if s.stats in ["min", "max"]
        ]
        if any(
            stats.params.method_type in ["Moving Average", "Gumbel"]
            for stats in parameterized_stats
        ):
            if self.time_scale_conversion is None:
                raise ValueError("Time scale conversion parameters must be specified!")
        return self

    @classmethod
    def from_file(cls, filename: pathlib.Path) -> BasePressureCaseConfig:
        yaml_vals = read_yaml(filename)
        cfg = cls(**yaml_vals)
        return cfg
