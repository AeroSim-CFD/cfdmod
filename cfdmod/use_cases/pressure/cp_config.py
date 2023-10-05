from typing import Literal

from pydantic import BaseModel, Field


class CpConfig(BaseModel):
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
    dynamic_pressure: dict[str, float] = Field(
        ...,
        title="Dynamic Pressure variables",
        description="Contains the data for calculating dynamic pressure",
    )
