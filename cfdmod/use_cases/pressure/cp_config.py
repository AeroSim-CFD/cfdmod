from __future__ import annotations

__all__ = ["CpConfig", "CpCaseConfig"]

import pathlib
from typing import Literal, Optional

from pydantic import BaseModel, Field, field_validator, model_validator

from cfdmod.api.configs.hashable import HashableConfig
from cfdmod.use_cases.pressure.extreme_values import TimeScaleParameters
from cfdmod.use_cases.pressure.statistics import BasicStatisticModel, ParameterizedStatisticModel
from cfdmod.utils import read_yaml


class CpConfig(HashableConfig):
    number_of_chunks: int = Field(
        1,
        title="Number of chunks",
        description="How many chunks the output time series will be split into",
        ge=1,
    )
    timestep_range: tuple[float, float] = Field(
        ...,
        title="Timestep Range",
        description="Interval between start and end steps to slice data",
    )
    reference_pressure: Literal["average", "instantaneous"] = Field(
        ...,
        title="Reference Pressure",
        description="Sets how to account for reference pressure effects."
        + "If set to average, static pressure signal will be averaged."
        + "If set to instantaneous, static pressure signal will be transient.",
    )
    U_H: float = Field(
        ...,
        title="Reference Flow Velocity",
        description="Value for reference Flow Velocity to calculate dynamic pressure",
    )
    U_H_correction_factor: float = Field(
        1,
        title="Reference Flow Velocity correction factor",
        description="Value for reference Flow Velocity correction factor multiplier",
    )
    statistics: list[BasicStatisticModel | ParameterizedStatisticModel] = Field(
        ...,
        title="List of statistics",
        description="List of statistics to calculate from pressure coefficient signal",
    )

    @field_validator("statistics", mode="before")
    def validate_statistics(cls, v):
        stats_types = [s["stats"] for s in v]
        if len(set(stats_types)) != len(stats_types):
            raise Exception("Duplicated statistics! It can only have one statistic of each type")
        if "mean_eq" in stats_types:
            if any(expected_s not in stats_types for expected_s in ["mean", "min", "max"]):
                raise Exception("Equivalent mean (mean_eq) requires mean, min and max statistics")
        validated_list = []
        for statistic in v:
            if "params" in statistic.keys():
                validated_list.append(ParameterizedStatisticModel(**statistic))
            else:
                validated_list.append(BasicStatisticModel(**statistic))
        return validated_list


class CpCaseConfig(BaseModel):
    pressure_coefficient: dict[str, CpConfig] = Field(
        ...,
        title="Pressure Coefficient configs",
        description="Dictionary with Pressure Coefficient configuration",
    )
    time_scale_conversion: Optional[TimeScaleParameters] = Field(
        None,
        title="Time scale conversion parameters",
        description="Parameters for converting time scale",
    )

    @model_validator(mode="after")
    def check_extreme_values_params(self) -> CpCaseConfig:
        parameterized_stats = [
            s
            for v in self.pressure_coefficient.values()
            for s in v.statistics
            if s.stats in ["min", "max"]
        ]
        if any(
            stats.params.method_type in ["Moving Average", "Gumbel"]
            for stats in parameterized_stats
        ):
            if self.time_scale_conversion is None:
                raise ValueError("Time scale conversion parameters must be specified!")
        return self

    @classmethod
    def from_file(cls, filename: pathlib.Path) -> CpCaseConfig:
        yaml_vals = read_yaml(filename)
        cfg = cls(**yaml_vals)
        return cfg
