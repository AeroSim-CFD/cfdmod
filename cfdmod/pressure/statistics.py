__all__ = ["Statistics"]

from typing import Literal, get_args

import numpy as np
from pydantic import BaseModel, Field, field_validator

Statistics = Literal["max", "min", "rms", "mean", "mean_eq", "skewness", "kurtosis"]
ExtremeMethods = Literal["Gumbel", "Peak", "Absolute", "Moving Average"]


class ExtremeAbsoluteParamsModel(BaseModel):
    method_type: Literal["Absolute"] = "Absolute"


class ExtremeGumbelParamsModel(BaseModel):
    method_type: Literal["Gumbel"] = "Gumbel"
    peak_duration: float
    event_duration: float
    n_subdivisions: int = 10
    non_exceedance_probability: float = Field(0.78, gt=0, lt=1)
    full_scale_U_H: float = Field(gt=0)
    full_scale_characteristic_length: float = Field(gt=0)

    @property
    def yR(self):
        if not hasattr(self, "_yR"):
            self._yR = -np.log(-np.log(self.non_exceedance_probability))
        return self._yR


class ExtremePeakParamsModel(BaseModel):
    method_type: Literal["Peak"] = "Peak"
    peak_factor: float


class ExtremeMovingAverageParamsModel(BaseModel):
    method_type: Literal["Moving Average"] = "Moving Average"
    window_size_interval: float = Field(gt=0)
    full_scale_U_H: float = Field(gt=0)
    full_scale_characteristic_length: float = Field(gt=0)


class MeanEquivalentParamsModel(BaseModel):
    scale_factor: float = Field(default=0.61, gt=0, le=1)


class BasicStatisticModel(BaseModel):
    stats: Statistics
    display_name: str = ""


StatisticsParamsModel = (
    MeanEquivalentParamsModel
    | ExtremeGumbelParamsModel
    | ExtremePeakParamsModel
    | ExtremeAbsoluteParamsModel
    | ExtremeMovingAverageParamsModel
)


class ParameterizedStatisticModel(BasicStatisticModel):
    params: StatisticsParamsModel

    @field_validator("params", mode="before")
    def validate_params(cls, v):
        validated_params = None
        if not isinstance(v, dict):
            # Already validated
            return v
        if "method_type" in v.keys():
            if v["method_type"] == "Gumbel":
                validated_params = ExtremeGumbelParamsModel(**v)
            elif v["method_type"] == "Peak":
                validated_params = ExtremePeakParamsModel(**v)
            elif v["method_type"] == "Absolute":
                validated_params = ExtremeAbsoluteParamsModel(**v)
            elif v["method_type"] == "Moving Average":
                validated_params = ExtremeMovingAverageParamsModel(**v)
            else:
                available_methods = get_args(ExtremeMethods)
                raise ValueError(
                    f"Unknown method {v['method_type']}, available methods are {available_methods}"
                )
        else:
            validated_params = MeanEquivalentParamsModel(**v)

        return validated_params
