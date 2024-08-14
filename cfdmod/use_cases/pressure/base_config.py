from __future__ import annotations

from pydantic import BaseModel, Field, field_validator

from cfdmod.use_cases.pressure.statistics import BasicStatisticModel, ParameterizedStatisticModel


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
