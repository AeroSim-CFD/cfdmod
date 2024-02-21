from __future__ import annotations

import pathlib
from enum import Enum
from typing import Literal, Optional

from pydantic import BaseModel, ConfigDict, Field, model_validator

from cfdmod.utils import read_yaml

PROJECTION_CASES = Literal["x_plus", "x_minus", "y_plus", "y_minus"]


class Projections(Enum):
    x_plus = ("x_plus", (0, -90, 0))
    x_minus = ("x_minus", (0, 90, 0))
    y_plus = ("y_plus", (-90, 0, 0))
    y_minus = ("y_minus", (90, 0, 0))


class CropConfig(BaseModel):
    width_ratio: float = Field(
        1,
        title="Crop width ratio",
        description="Ratio for cropping the rendered image",
        gt=0,
        le=1,
    )
    height_ratio: float = Field(
        1,
        title="Crop height ratio",
        description="Ratio for cropping the rendered image",
        gt=0,
        le=1,
    )
    watermark_path: Optional[str] = Field(
        None, title="Watermark path", description="Path for the image to be used as watermark"
    )


class CameraConfig(BaseModel):
    zoom: float = Field(1, title="Camera zoom", gt=0)
    offset_position: tuple[float, float, float] = Field(
        (0, 0, 0),
        title="Camera position offset",
        description="Value for offsetting the camera position",
    )
    view_up: tuple[float, float, float] = Field(
        (1, 0, 0), title="Camera view up", description="Camera view up direction vector"
    )
    window_size: tuple[int, int] = Field(
        (800, 800), title="Window size", description="Height and width of the rendering window"
    )
    crop: CropConfig = Field(
        CropConfig(), title="Crop configuration", description="Parameters for cropping"
    )


class ImageConfig(BaseModel):
    scalar_label: str = Field(
        ..., title="Scalar label", description="Label of the scalar to set active on the snapshot"
    )
    image_label: str = Field(..., title="Image label", description="Label of the output image")


class ColormapConfig(BaseModel):
    n_divs: int = Field(
        None, title="Number of divisions", description="Colormap divisions", ge=3, le=15
    )
    target_step: float = Field(None, title="Target step", description="Colormap target step", gt=0)

    def get_colormap_divs(self, scalar_range: tuple[float, float]) -> int:
        if self.n_divs != None:
            return self.n_divs
        else:
            divs = round((scalar_range[1] - scalar_range[0]) / self.target_step)
            return divs

    @model_validator(mode="after")
    def exclusive_props(self) -> ColormapConfig:
        if self.n_divs != None and self.target_step != None:
            raise ValueError("Cannot set both num_steps and target_step")
        return self


class ProjectionConfig(BaseModel):
    model_config = ConfigDict(use_enum_values=True)
    offset: float = Field(
        10,
        title="Offset value",
        description="Value for offsetting each projection from the center projection",
        ge=0,
    )
    axis: list[PROJECTION_CASES] = Field(
        ["x_plus", "x_minus", "y_plus", "y_minus"],
        title="Projection axis",
        description="Select which axis are included in the projections.",
    )
    rotation: tuple[float, float, float] = Field(
        (0, 0, 0),
        title="Rotation vector",
        description="Vector to rotate the body for setting up the projection",
    )


class SnapshotConfig(BaseModel):
    images: list[ImageConfig] = Field(
        ..., title="Image list", description="List of images to generate snapshots"
    )
    projection: ProjectionConfig = Field(
        ..., title="Projection configuration", description="Parameters for the projections"
    )
    colormap: ColormapConfig = Field(
        ..., title="Colormap configuration", description="Parameters for colormap"
    )
    camera: CameraConfig = Field(
        ..., title="Camera configuration", description="Parameters for setting up the camera"
    )

    @classmethod
    def from_file(cls, filename: pathlib.Path) -> SnapshotConfig:
        yaml_vals = read_yaml(filename)
        cfg = cls(**yaml_vals)

        return cfg
