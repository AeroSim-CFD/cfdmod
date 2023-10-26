from __future__ import annotations

__all__ = ["BodyConfig"]

from pydantic import BaseModel, Field, field_validator

from cfdmod.use_cases.pressure.zoning.zoning_model import ZoningModel


class BodyConfig(BaseModel):
    surfaces: list[str] = Field(
        ..., title="Body's surfaces", description="List of surfaces that compose the body"
    )
    sub_bodies: ZoningModel = Field(
        ZoningModel(
            x_intervals=[float("-inf"), float("inf")],
            y_intervals=[float("-inf"), float("inf")],
            z_intervals=[float("-inf"), float("inf")],
        ),
        title="Sub body intervals",
        description="Definition of the intervals that will section the body into sub-bodies",
    )

    @field_validator("surfaces")
    def validate_surface_list(cls, v):
        if len(v) != len(set(v)):
            raise Exception("Invalid exceptions surface list, names must not repeat")
        return v
