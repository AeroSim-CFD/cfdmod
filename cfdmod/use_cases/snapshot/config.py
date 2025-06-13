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


class TransformationConfig(BaseModel):
    translate: tuple[float, float, float] = Field(
        (0, 0, 0),
        title="Translate vector",
        description="Vector to representing the translation",
    )
    rotate: tuple[float, float, float] = Field(
        (0, 0, 0),
        title="Rotate vector",
        description="Vector to representing the rotation",
    )
    scale: tuple[float, float, float] = Field(
        (1, 1, 1),
        title="Scale vector",
        description="Vector to representing the scale",
    )


class LegendConfig(BaseModel):
    label: str = Field(..., title="Legend name", description="The name of the legend in the image")
    range: tuple[float, float] = Field(
        ..., title="Legend range values", description="Range of values in legend"
    )
    n_divs: int = Field(
        ..., title="Number of divisions", description="Number of divisions in legend"
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
    name: str = Field(..., title="Image label", description="Label of the output image")
    legend_config: LegendConfig = Field(
        ..., title="Legend configuration", description="Image legend configuration"
    )
    projections: dict[str, PartialProjectionConfig] = Field(
        ..., title="Projections", description="Projections in the image"
    )


class ColormapConfig(BaseModel):
    style: str = "contour"
    n_divs: int = Field(
        None, title="Number of divisions", description="Colormap divisions", ge=3, le=15
    )
    target_step: float = Field(None, title="Target step", description="Colormap target step", gt=0)

    def get_colormap_divs(self, scalar_range: tuple[float, float]) -> int:
        if self.n_divs is not None:
            return self.n_divs
        else:
            divs = round((scalar_range[1] - scalar_range[0]) / self.target_step)
            return divs

    @model_validator(mode="after")
    def exclusive_props(self) -> ColormapConfig:
        if self.n_divs is not None and self.target_step is not None:
            raise ValueError("Cannot set both num_steps and target_step")
        return self


class ProjectionConfig(BaseModel):
    file_path: pathlib.Path = Field(
        ...,
        title="Polydata file path",
        description="Path to the polydata file",
    )
    scalar: str = Field(
        ...,
        title="Scalar field",
        description="Label of the scalar to set active on the projection",
    )
    clip_box: TransformationConfig = Field(
        ...,
        title="ClipBox configuration",
        description="Parameters for clipbox",
    )
    transformation: TransformationConfig = Field(
        ...,
        title="Transformation components",
        description="Parameters to represent the transformation of the body in the projection",
    )


class PartialProjectionConfig(BaseModel):
    file_path: pathlib.Path = Field(
        ...,
        title="Polydata file path",
        description="Path to the polydata file",
    )
    scalar: str = Field(
        ...,
        title="Scalar field",
        description="Label of the scalar to set active on the projection",
    )


class SnapshotConfig(BaseModel):
    name: str = Field(..., title="Snapshot name", description="The name of the snapshot")
    legend_config: LegendConfig = Field(
        ..., title="Legend configuration", description="Image legend configuration"
    )
    projections: dict[str, ProjectionConfig] = Field(
        ..., title="Projections configuration", description="Parameters for the projections"
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
