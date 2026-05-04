"""Statistics spec types.

Pydantic models declaring the catalogue of statistics callers can ask
for (mean / rms / extremes / mean-equivalent) and the parameters each
extreme-value method consumes (Gumbel, Peak, Absolute).

These types used to live in :mod:`cfdmod.pressure.parameters`; they
were moved here so the statistics package stops being a downstream
consumer of pressure types and the cycle (statistics -> pressure ->
statistics) is broken. ``cfdmod.pressure.parameters`` re-exports them
for back-compat.
"""

from __future__ import annotations

__all__ = [
    "Statistics",
    "ExtremeMethods",
    "BasicStatisticModel",
    "ParameterizedStatisticModel",
    "StatisticsParamsModel",
    "ExtremeAbsoluteParamsModel",
    "ExtremeGumbelParamsModel",
    "ExtremePeakParamsModel",
    "MeanEquivalentParamsModel",
]

from typing import Annotated, Literal, get_args

import numpy as np
from pydantic import BaseModel, Field, field_validator

Statistics = Literal["max", "min", "rms", "mean", "mean_eq", "skewness", "kurtosis"]
ExtremeMethods = Literal["Gumbel", "Peak", "Absolute"]


class ExtremeAbsoluteParamsModel(BaseModel):
    method_type: Literal["Absolute"] = "Absolute"


class ExtremeGumbelParamsModel(BaseModel):
    method_type: Literal["Gumbel"] = "Gumbel"
    peak_duration: float
    event_duration: float
    n_subdivisions: int = 10
    non_exceedance_probability: Annotated[float, Field(0.78, gt=0, lt=1)]
    # Optional in Cf/Cm: when omitted, the runner inherits from the Cp config
    # used to produce the input cp_h5 (simul_U_H / simul_characteristic_length
    # are persisted in /processing_metadata).
    full_scale_U_H: Annotated[float | None, Field(default=None, gt=0)]
    full_scale_characteristic_length: Annotated[float | None, Field(default=None, gt=0)]

    @property
    def yR(self):
        if not hasattr(self, "_yR"):
            self._yR = -np.log(-np.log(self.non_exceedance_probability))
        return self._yR


class ExtremePeakParamsModel(BaseModel):
    method_type: Literal["Peak"] = "Peak"
    peak_factor: float


class MeanEquivalentParamsModel(BaseModel):
    scale_factor: Annotated[float, Field(default=0.61, gt=0, le=1)]


StatisticsParamsModel = (
    MeanEquivalentParamsModel
    | ExtremeGumbelParamsModel
    | ExtremePeakParamsModel
    | ExtremeAbsoluteParamsModel
)


class BasicStatisticModel(BaseModel):
    stats: Statistics
    display_name: str = ""


class ParameterizedStatisticModel(BasicStatisticModel):
    params: StatisticsParamsModel

    @field_validator("params", mode="before")
    def validate_params(cls, v):
        if not isinstance(v, dict):
            return v
        if "method_type" in v:
            if v["method_type"] == "Gumbel":
                return ExtremeGumbelParamsModel(**v)
            elif v["method_type"] == "Peak":
                return ExtremePeakParamsModel(**v)
            elif v["method_type"] == "Absolute":
                return ExtremeAbsoluteParamsModel(**v)
            else:
                available = get_args(ExtremeMethods)
                raise ValueError(
                    f"Unknown method {v['method_type']}, available: {available}"
                )
        return MeanEquivalentParamsModel(**v)
