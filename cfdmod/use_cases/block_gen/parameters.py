import pathlib
from enum import Enum

from pydantic import BaseModel, Field

from cfdmod.utils import read_yaml

__all__ = [
    "OffsetDirection",
    "GenerationParams",
    "BlockParams",
    "SpacingParams",
]


class OffsetDirection(str, Enum):
    """Define the offset direction for block lines"""

    x = "x"
    y = "y"


class SpacingParams(BaseModel):
    spacing_x: float = Field(
        ...,
        name="X spacing",
        description="Block line X spacing",
        gt=0,
    )
    spacing_y: float = Field(
        ...,
        name="Y spacing",
        description="Block line Y spacing",
        gt=0,
    )
    line_offset: float = Field(
        ...,
        name="Line offset",
        description="Offset percentage between each block line",
        ge=0,
    )
    is_abs: bool = Field(
        True,
        description="Flag to determine whether the line offset is absolute or relative to the spacing (percentage)",
    )
    offset_direction: OffsetDirection = Field(
        OffsetDirection.y,
        name="Offset Direction",
        description="Direction which the blocks should be offseted to",
    )


class BlockParams(BaseModel):
    height: float = Field(
        ...,
        name="Block height",
        description="Size of the generated blocks in Z axis",
        gt=0,
    )
    width: float = Field(
        ...,
        name="Block width",
        description="Size of the generated blocks in Y axis",
        gt=0,
    )
    length: float = Field(
        ...,
        name="Block length",
        description="Size of the generated blocks in X axis",
        gt=0,
    )


class GenerationParams(BaseModel):
    N_blocks_x: int = Field(
        ...,
        name="Number of blocks in X",
        description="Defines the number of blocks in the X axis",
        gt=0,
    )
    N_blocks_y: int = Field(
        ...,
        name="Number of blocks in Y",
        description="Defines the number of blocks in the Y axis",
        gt=0,
    )
    block_params: BlockParams = Field(
        ..., name="Block parameters", description="Object with block parameters"
    )
    spacing_params: SpacingParams = Field(
        ..., name="Spacing parameters", description="Object with spacing parameters"
    )

    def calculate_spacing(self, direction: OffsetDirection) -> float:
        offset_size = (
            self.block_params.length + self.spacing_params.spacing_x
            if self.spacing_params.offset_direction == "x"
            else self.block_params.width + self.spacing_params.spacing_y
        )
        line_offset = (
            self.spacing_params.line_offset
            if self.spacing_params.is_abs
            else self.spacing_params.line_offset * offset_size
        )
        return line_offset

    @property
    def perpendicular_direction(self) -> OffsetDirection:
        return (
            OffsetDirection.x
            if self.spacing_params.offset_direction == OffsetDirection.y
            else OffsetDirection.y
        )

    @classmethod
    def from_file(cls, file_path: pathlib.Path):
        if file_path.exists():
            yaml_vals = read_yaml(file_path)
            params = cls(**yaml_vals)
            return params
        else:
            raise Exception(f"Unable to read yaml. Filename {file_path.name} does not exists")
