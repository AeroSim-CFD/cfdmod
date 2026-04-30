from __future__ import annotations

import pathlib
from typing import Annotated, Union

from pydantic import BaseModel, Field, field_validator, model_validator

from cfdmod.utils import read_yaml


class ImageConfig(BaseModel):
    name: Annotated[str, Field(..., title="Image label", description="Label of the output image")]
    legend_config: Annotated[
        LegendConfig,
        Field(..., title="Legend configuration", description="Image legend configuration"),
    ]
    projections: Annotated[
        dict[str, ProjectionConfig],
        Field(..., title="Projections", description="Projections in the image"),
    ]


class CropConfig(BaseModel):
    width_ratio: Annotated[
        float,
        Field(
            1,
            title="Crop width ratio",
            description="Ratio for cropping the rendered image",
            gt=0,
            le=1,
        ),
    ]
    height_ratio: Annotated[
        float,
        Field(
            1,
            title="Crop height ratio",
            description="Ratio for cropping the rendered image",
            gt=0,
            le=1,
        ),
    ]


class OverlayImageConfig(BaseModel):
    image_path: Annotated[
        pathlib.Path,
        Field(
            ...,
            title="overlay image path",
            description="Path for the image to be overlayed on the snapshot",
        ),
    ]
    position: Annotated[
        tuple[float, float],
        Field(
            (0, 0),
            title="Position of image overlay",
            description="Coordinates where the image will be overlayed",
        ),
    ]
    angle: Annotated[
        float,
        Field(
            0,
            title="Image rotation angle",
            description="Angle of rotation of image",
        ),
    ]
    scale: Annotated[
        float,
        Field(
            1,
            title="Image reescale",
            description="Scale to be applied on the image before overlaying",
        ),
    ]
    transparency: Annotated[
        float,
        Field(
            0,
            title="Image transparency",
            description="Image transparency to be applied before overlaying. 1=fully transparent",
        ),
    ]

    @field_validator("image_path", mode="before")
    def normalize_path(cls, v: str | pathlib.Path) -> pathlib.Path:
        if isinstance(v, str):
            return pathlib.Path(v)
        if isinstance(v, pathlib.Path):
            return v
        raise ValueError("Image path must be a string or pathlib.Path")


class OverlayTextConfig(BaseModel):
    text: Annotated[
        str,
        Field(
            ...,
            title="overlay image path",
            description="Path for the image to be overlayed on the snapshot",
        ),
    ]
    position: Annotated[
        tuple[float, float],
        Field(
            (0, 0),
            title="Position of image overlay",
            description="Coordinates where the image will be overlayed",
        ),
    ]
    angle: Annotated[
        float,
        Field(
            0,
            title="Text rotation angle",
            description="Angle of rotation of text in z axis",
        ),
    ]
    font_size: Annotated[
        float,
        Field(
            12,
            title="Font size",
            description="Size of the font of the text to be overlayed",
        ),
    ]


class TransformationConfig(BaseModel):
    translate: Annotated[
        tuple[float, float, float],
        Field(
            (0, 0, 0),
            title="Translate vector",
            description="Vector representing the translation",
        ),
    ]
    rotate: Annotated[
        tuple[float, float, float],
        Field(
            (0, 0, 0),
            title="Rotate vector",
            description="Vector representing the rotation",
        ),
    ]
    scale: Annotated[
        tuple[float, float, float],
        Field(
            (1, 1, 1),
            title="Scale vector",
            description="Vector representing the scale",
        ),
    ]
    fixed_point: Annotated[
        tuple[float, float, float] | None,
        Field(
            None,
            title="Fixed point vector",
            description="Vector representing the origin point of scale and rotation",
        ),
    ]


class LegendConfig(BaseModel):
    label: Annotated[
        str, Field(..., title="Legend name", description="The name of the legend in the image")
    ]
    range: Annotated[
        tuple[float, float] | None,
        Field(None, title="Legend range values", description="Range of values in legend"),
    ]
    n_divs: Annotated[
        int | None,
        Field(None, title="Number of divisions", description="Number of divisions in legend"),
    ]
    custom_colorbar: Annotated[
        ColormapConfig | None,
        Field(
            None,
            title="Custom colorbar config",
            description="Manual config of colorbar. Requires assignement of exact divisions and colors to use.",
        ),
    ]

    @model_validator(mode="before")
    @classmethod
    def at_least_one_valid_option_was_chosen(cls, data) -> LegendConfig:
        if "custom_colorbar" not in data.keys():
            if "range" not in data.keys() or "n_divs" not in data.keys():
                ValueError(
                    "At least one of the two must be set: (range + n_divs) or (custom_colorbar)."
                )
            return data
        else:
            data["range"] = (
                data["custom_colorbar"]["value_edges"][0],
                data["custom_colorbar"]["value_edges"][-1],
            )
            data["n_divs"] = len(data["custom_colorbar"]["colors"])
            return data


class ColormapConfig(BaseModel):
    value_edges: Annotated[
        list[float],
        Field(
            ...,
            title="Rotate vector",
            description="Vector representing the rotation",
        ),
    ]
    colors: Annotated[
        list[str],
        Field(
            ...,
            title="List of colors",
            description="List of custom colors in hex notation",
        ),
    ]


class CameraConfig(BaseModel):
    zoom: Annotated[float, Field(1, title="Camera zoom", gt=0)]
    offset_position: Annotated[
        tuple[float, float],
        Field(
            (0, 0),
            title="Camera position offset",
            description="Value for offsetting the camera position",
        ),
    ]
    view_up: Annotated[
        tuple[float, float, float],
        Field((0, 1, 0), title="Camera view up", description="Camera view up direction vector"),
    ]
    window_size: Annotated[
        tuple[int, int],
        Field(
            (800, 800), title="Window size", description="Height and width of the rendering window"
        ),
    ]


class ValueTagsConfig(BaseModel):
    spacing: Annotated[tuple[float, float] | None, Field(None, description="Spacing (x, y)")]
    padding: Annotated[
        tuple[float, float, float, float] | None,
        Field(None, description="Padding (left, right, bottom, top)"),
    ]
    x: Annotated[
        list[float] | None,
        Field(
            None, description="Exact positions in x. Relative to bounding box of transformed mesh."
        ),
    ]
    y: Annotated[
        list[float] | None,
        Field(
            None, description="Exact positions in y. Relative to bounding box of transformed mesh."
        ),
    ]

    z_offset: Annotated[
        float,
        Field(
            default=0,
            title="Values tag search plane z offset",
            description="Negative z offset for plane where closest points in mesh will be searched",
            gt=0,
        ),
    ]

    decimal_places: Annotated[
        int,
        Field(
            default=2,
            title="Decimal places",
            description="Precision of results to be marked on tags",
            gt=0,
        ),
    ]

    @model_validator(mode="before")
    @classmethod
    def at_least_one_valid_option_was_chosen(cls, data) -> LegendConfig:
        cols = data.keys()
        if ("x" in cols) or (("y" in cols)) and not (("x" in cols) and (("y" in cols))):
            ValueError("Exact position set for x or y, but one is empty. Both must be specified.")
        elif ("x" in cols) and (("y" in cols)):
            # exact has precedence over floating
            data["spacing"] = None
            data["padding"] = None
        elif (
            ("spacing" in cols)
            or (("padding" in cols))
            and not (("spacing" in cols) and (("padding" in cols)))
        ):
            ValueError(
                "Floating position set for spacing or padding, but one is empty. Both must be specified."
            )
        return data

    @field_validator("spacing", mode="before")
    def normalize_spacing(cls, v: Union[float, tuple] | None) -> tuple[float, float]:
        if v is None:
            return None
        if isinstance(v, (int, float)):
            return (float(v), float(v))
        if isinstance(v, tuple) or isinstance(v, list) and len(v) == 2:
            return tuple(map(float, v))
        raise ValueError("spacing must be a float or a 2-tuple of floats")

    @field_validator("padding", mode="before")
    def normalize_padding(cls, v: Union[float, tuple] | None) -> tuple[float, float, float, float]:
        if v is None:
            return None
        if isinstance(v, (int, float)):
            return (float(v), float(v), float(v), float(v))
        if isinstance(v, list) or isinstance(v, tuple):
            if len(v) == 2:
                return (float(v[0]), float(v[0]), float(v[1]), float(v[1]))
            if len(v) == 4:
                return tuple(map(float, v))
        raise ValueError("padding must be a float, a 2-tuple, or a 4-tuple of floats")


class ProjectionConfig(BaseModel):
    file_path: Annotated[
        pathlib.Path,
        Field(
            ...,
            title="Polydata file path",
            description="Path to the polydata file",
        ),
    ]
    scalar: Annotated[
        str | None,
        Field(
            None,
            title="Scalar field",
            description="Label of the scalar to set active on the projection",
        ),
    ]
    cell_data_to_point_data: Annotated[
        bool,
        Field(
            True,
            title="Apply cell_data_to_point_data and contour filters",
            description="True gives a smooth apearance, False preserves better the separations of Ce and Cf.",
        ),
    ]
    values_tag_config: Annotated[ValueTagsConfig | None, Field(None, title="", description="")]
    clip_box: Annotated[
        TransformationConfig | None,
        Field(
            None,
            title="ClipBox configuration",
            description="Parameters for clipbox",
        ),
    ]
    transformation: Annotated[
        TransformationConfig,
        Field(
            TransformationConfig(),
            title="Transformation components",
            description="Parameters to represent the transformation of the body in the projection",
        ),
    ]


class SnapshotConfig(BaseModel):
    projections: Annotated[
        dict[str, ProjectionConfig],
        Field(
            ..., title="Labels configuration", description="Parameters for the projection labels"
        ),
    ]
    images_overlay: Annotated[
        list[OverlayImageConfig],
        Field(
            [],
            title="Images to overlay",
            description="List of images to be overlayed on the snapshot",
        ),
    ]
    text_overlay: Annotated[
        list[OverlayTextConfig],
        Field(
            [],
            title="Text to overlay",
            description="List of textes to be overlayed on the snapshot",
        ),
    ]
    legend_config: Annotated[
        LegendConfig,
        Field(..., title="Legend configuration", description="Image legend configuration"),
    ]
    camera: Annotated[
        CameraConfig | None,
        Field(
            ..., title="Camera configuration", description="Parameters for setting up the camera"
        ),
    ]
    image_crop: Annotated[
        CropConfig | None,
        Field(None, title="Crop configuration", description="Parameters for cropping"),
    ]

    @classmethod
    def from_file(cls, filename: str | pathlib.Path) -> SnapshotConfig:
        if isinstance(filename, str):
            filename = pathlib.Path(filename)
        yaml_vals = read_yaml(filename)
        cfg = cls(**yaml_vals)

        return cfg
