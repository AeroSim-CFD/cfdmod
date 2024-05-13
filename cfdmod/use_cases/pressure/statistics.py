__all__ = ["Statistics"]

from typing import Literal

from pydantic import BaseModel, Field, field_validator

Statistics = Literal["max", "min", "rms", "mean", "mean_eq", "skewness", "kurtosis"]


class ExtremeAbsoluteParamsModel(BaseModel):
    method_type: Literal["Absolute"] = "Absolute"


class ExtremeGumbelParamsModel(BaseModel):
    method_type: Literal["Gumbel"] = "Gumbel"
    peak_duration: float
    event_duration: float
    n_subdivisions: int
    non_exceedance_probability: float


class ExtremePeakParamsModel(BaseModel):
    method_type: Literal["Peak"] = "Peak"
    peak_factor: float


class ExtremeMovingAverageParamsModel(BaseModel):
    method_type: Literal["Moving Average"] = "Moving Average"
    window_size_real_scale: float = Field(gt=0)


class MeanEquivalentParamsModel(BaseModel):
    time_scale_factor: float = Field(default=0.61, gt=0, le=1)


class BasicStatisticModel(BaseModel):
    stats: Statistics


class ParameterizedStatisticModel(BasicStatisticModel):
    params: (
        MeanEquivalentParamsModel
        | ExtremeGumbelParamsModel
        | ExtremePeakParamsModel
        | ExtremeAbsoluteParamsModel
        | ExtremeMovingAverageParamsModel
    )

    @field_validator("params", mode="before")
    def validate_params(cls, v):
        validated_params = None
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
                available_methods = ["Gumbel", "Peak", "Absolute", "Moving Average"]
                raise ValueError(
                    f"Unknown method {v['method_type']}, available methods are {available_methods}"
                )
        else:
            validated_params = MeanEquivalentParamsModel(**v)

        return validated_params
