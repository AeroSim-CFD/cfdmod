__all__ = ["Body"]

from pydantic import BaseModel, Field, validator


class SubBodyIntervals(BaseModel):
    x_intervals: list[float] = Field(
        [],
        title="X intervals list",
        description="Values for the X axis intervals list, it must be unique",
    )
    y_intervals: list[float] = Field(
        [],
        title="Y intervals list",
        description="Values for the Y axis intervals list, it must be unique",
    )
    z_intervals: list[float] = Field(
        [],
        title="Z intervals list",
        description="Values for the Z axis intervals list, it must be unique",
    )

    @validator("x_intervals", "y_intervals", "z_intervals", always=True)
    def validate_intervals(cls, v):
        if len(v) > 1:
            for i in range(len(v) - 1):
                if v[i] >= v[i + 1]:
                    raise Exception("Interval must have ascending order")
        return v


class Body(BaseModel):
    surfaces: list[str] = Field(
        ..., title="Body's surfaces", description="List of surfaces that compose the body"
    )
    sub_bodies: SubBodyIntervals = Field(
        SubBodyIntervals(x_intervals=[], y_intervals=[], z_intervals=[]),
        title="Sub body intervals",
        description="Definition of the intervals that will section the body into sub-bodies",
    )

    @validator("surfaces", always=True)
    def validate_surface_list(cls, v):
        if len(v) != len(set(v)):
            raise Exception("Invalid exceptions surface list, names must not repeat")
        return v
