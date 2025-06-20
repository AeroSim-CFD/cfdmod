from __future__ import annotations

import pathlib
from enum import Enum
from typing import Literal, Optional, Union

from pydantic import BaseModel, Field, field_validator, model_validator

from cfdmod.utils import read_yaml

class ImageConfig(BaseModel):
    name: str = Field(..., title="Image label", description="Label of the output image")
    legend_config: LegendConfig = Field(
        ..., title="Legend configuration", description="Image legend configuration"
    )
    projections: dict[str, ProjectionConfig] = Field(
        ..., title="Projections", description="Projections in the image"
    )

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

class OverlayImageConfig(BaseModel):
    image_path: pathlib.Path = Field(
        ..., title="overlay image path", description="Path for the image to be overlayed on the snapshot"
    )
    position: tuple[float,float] = Field(        
        (0, 0),
        title="Position of image overlay",
        description="Coordinates where the image will be overlayed",
    )
    angle: float = Field(        
        0,
        title="Image rotation angle",
        description="Angle of rotation of image",
    )
    scale: float = Field(        
        1,
        title="Image reescale",
        description="Scale to be applied on the image before overlaying",
    )
    transparency: float = Field(
        0,
        title="Image transparency",
        description="Image transparency to be applied before overlaying. 1=fully transparent",
    )
    
    @field_validator("image_path", mode="before")
    def normalize_path(cls, v: str|pathlib.Path) -> pathlib.Path:
        if isinstance(v, str):
            return pathlib.Path(v)
        if isinstance(v, pathlib.Path):
            return v
        raise ValueError("Image path must be a string or pathlib.Path")

class OverlayTextConfig(BaseModel):
    text: str = Field(
        ..., title="overlay image path", description="Path for the image to be overlayed on the snapshot"
    )
    position: tuple[float,float] = Field(        
        (0, 0),
        title="Position of image overlay",
        description="Coordinates where the image will be overlayed",
    )
    angle: float = Field(        
        0,
        title="Text rotation angle",
        description="Angle of rotation of text in z axis",
    )
    font_size: float = Field(        
        12,
        title="Font size",
        description="Size of the font of the text to be overlayed",
    )
    
class TransformationConfig(BaseModel):
    translate: tuple[float, float, float] = Field(
        (0, 0, 0),
        title="Translate vector",
        description="Vector representing the translation",
    )
    rotate: tuple[float, float, float] = Field(
        (0, 0, 0),
        title="Rotate vector",
        description="Vector representing the rotation",
    )
    scale: tuple[float, float, float] = Field(
        (1, 1, 1),
        title="Scale vector",
        description="Vector representing the scale",
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
    offset_position: tuple[float, float] = Field(
        (0, 0),
        title="Camera position offset",
        description="Value for offsetting the camera position",
    )
    view_up: tuple[float, float, float] = Field(
        (0, 1, 0), title="Camera view up", description="Camera view up direction vector"
    )
    window_size: tuple[int, int] = Field(
        (800, 800), title="Window size", description="Height and width of the rendering window"
    )

class ValueTagsConfig(BaseModel):
    spacing: tuple[float, float] = Field(..., description="Spacing (x, y)")
    padding: tuple[float, float, float, float] = Field(
        ..., description="Padding (left, right, bottom, top)"
    )
    z_offset: float = Field(default=0, title="Negative z offset for plane where closest points in mesh will be searched", gt=0)

    @field_validator("spacing", mode="before")
    def normalize_spacing(cls, v: Union[float, tuple]) -> tuple[float, float]:
        if isinstance(v, (int, float)):
            return (float(v), float(v))
        if isinstance(v, tuple) and len(v) == 2:
            return tuple(map(float, v))
        raise ValueError("spacing must be a float or a 2-tuple of floats")

    @field_validator("padding", mode="before")
    def normalize_padding(cls, v: Union[float, tuple]) -> tuple[float, float, float, float]:
        if isinstance(v, (int, float)):
            return (float(v), float(v), float(v), float(v))
        if isinstance(v, list) or isinstance(v, tuple):
            if len(v) == 2:
                return (float(v[0]), float(v[0]), float(v[1]), float(v[1]))
            if len(v) == 4:
                return tuple(map(float, v))
        raise ValueError("padding must be a float, a 2-tuple, or a 4-tuple of floats")


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
        None,
        title="Scalar field",
        description="Label of the scalar to set active on the projection",
    )
    values_tag_config: ValueTagsConfig | None = Field(None, title="", description="")
    clip_box: TransformationConfig = Field(
        None,
        title="ClipBox configuration",
        description="Parameters for clipbox",
    )
    transformation: TransformationConfig = Field(
        default_factory=TransformationConfig,
        title="Transformation components",
        description="Parameters to represent the transformation of the body in the projection",
    )


class SnapshotConfig(BaseModel):
    projections: dict[str, ProjectionConfig] = Field(
        ..., title="Labels configuration", description="Parameters for the projection labels"
    )
    images_overlay: list[OverlayImageConfig] = Field(
        None, title="Images to overlay", description="List of images to be overlayed on the snapshot"
    )
    text_overlay: list[OverlayTextConfig] = Field(
        None, title="Text to overlay", description="List of textes to be overlayed on the snapshot"
    )
    legend_config: LegendConfig = Field(
        ..., title="Legend configuration", description="Image legend configuration"
    )
    colormap: ColormapConfig = Field(
        ..., title="Colormap configuration", description="Parameters for colormap"
    )
    camera: CameraConfig = Field(
        ..., title="Camera configuration", description="Parameters for setting up the camera"
    )
    image_crop: CropConfig = Field(
        None, title="Crop configuration", description="Parameters for cropping"
    )


    @classmethod
    def from_file(cls, filename: str|pathlib.Path) -> SnapshotConfig:
        if isinstance(filename, str):
            filename = pathlib.Path(filename)
        yaml_vals = read_yaml(filename)
        cfg = cls(**yaml_vals)

        return cfg
