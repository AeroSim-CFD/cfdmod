from __future__ import annotations

import pathlib
from enum import Enum

from pydantic import BaseModel, Field, field_validator, model_validator

from cfdmod.utils import read_yaml


class Projections(Enum):
    x_plus = (0, -90, 0)
    x_minus = (0, 90, 0)
    y_plus = (-90, 0, 0)
    y_minus = (90, 0, 0)


class CameraParameters:
    zoom: float
    offset_position: tuple[float, float, float]
    view_up: tuple[float, float, float]
    window_size: tuple[int, int]


class ImageConfig(BaseModel):
    scalar_label: str = Field(
        ..., title="Scalar label", description="Label of the scalar to set active on the snapshot"
    )
    image_label: str = Field(..., title="Image label", description="Label of the output image")


class ProjectionConfig(BaseModel):
    offset: float = Field(
        ...,
        title="Offset value",
        description="Value for offsetting each projection from the center projection",
        ge=0,
    )
    axis: list[Projections] = Field()
    rotation: tuple[float, float, float]


class SnapshotConfig(BaseModel):
    images: list[ImageConfig] = Field(
        ..., title="Image list", description="List of images to generate snapshots"
    )

    @classmethod
    def from_file(cls, filename: pathlib.Path) -> SnapshotConfig:
        yaml_vals = read_yaml(filename)
        cfg = cls(**yaml_vals)

        return cfg
