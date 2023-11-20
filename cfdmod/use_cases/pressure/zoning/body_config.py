from __future__ import annotations

__all__ = ["BodyConfig"]

from pydantic import BaseModel, Field, field_validator


class BodyConfig(BaseModel):
    surfaces: list[str] = Field(
        ..., title="Body's surfaces", description="List of surfaces that compose the body"
    )

    @field_validator("surfaces")
    def validate_surface_list(cls, v):
        if len(v) != len(set(v)):
            raise Exception("Invalid exceptions surface list, names must not repeat")
        return v
