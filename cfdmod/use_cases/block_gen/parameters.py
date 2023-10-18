import pathlib
from typing import Annotated, Literal

from pydantic import BaseModel, Field

from cfdmod.utils import read_yaml

__all__ = [
    "OffsetDirection",
    "GenerationParams",
    "BlockParams",
    "SpacingParams",
]


OffsetDirection = Annotated[
    Literal["x", "y"], Field(description="""Define the offset direction for block lines""")
]


class SpacingParams(BaseModel):
    spacing: tuple[float, float] = Field(
        ...,
        title="Spacing values",
        description="Spacing values in X axis (index 0) and Y axis (index 1)."
        + "The spacing between each line is calculated with the spacing value "
        + "plus the size of the block in respective direction",
    )
    line_offset: float = Field(
        ...,
        title="Line offset",
        description="Offset percentage between each block line",
        ge=0,
    )
    offset_direction: OffsetDirection = Field(
        "y",
        title="Offset Direction",
        description="Direction which the blocks should be offseted to",
    )


class BlockParams(BaseModel):
    height: float = Field(
        ...,
        title="Block height",
        description="Size of the generated blocks in Z axis",
        gt=0,
    )
    width: float = Field(
        ...,
        title="Block width",
        description="Size of the generated blocks in Y axis",
        gt=0,
    )
    length: float = Field(
        ...,
        title="Block length",
        description="Size of the generated blocks in X axis",
        gt=0,
    )


class GenerationParams(BaseModel):
    N_blocks_x: int = Field(
        ...,
        title="Number of blocks in X",
        description="Defines the number of blocks in the X axis",
        gt=0,
    )
    N_blocks_y: int = Field(
        ...,
        title="Number of blocks in Y",
        description="Defines the number of blocks in the Y axis",
        gt=0,
    )
    block_params: BlockParams = Field(
        ..., title="Block parameters", description="Object with block parameters"
    )
    spacing_params: SpacingParams = Field(
        ..., title="Spacing parameters", description="Object with spacing parameters"
    )

    @property
    def single_line_blocks(self) -> int:
        """Calculates the number of blocks in a single line based on the offset direction

        Returns:
            int: Number of repetitions applied to a block to form a row
        """
        match self.spacing_params.offset_direction:
            case "x":
                return self.N_blocks_x - 1
            case "y":
                return self.N_blocks_y - 1

    @property
    def single_line_spacing(self) -> float:
        """Calculates the single line spacing based on the offset direction

        Returns:
            float: Value for spacing the blocks in a single row
        """
        match self.spacing_params.offset_direction:
            case "x":
                return self.spacing_params.spacing[0] + self.block_params.length
            case "y":
                return self.spacing_params.spacing[1] + self.block_params.width

    @property
    def multi_line_blocks(self) -> int:
        """Calculates the number of rows to be replicated based on the offset direction

        Returns:
            int: Number of repetitions applied to a row of blocks
        """
        match self.spacing_params.offset_direction:
            case "x":
                return self.N_blocks_y - 1
            case "y":
                return self.N_blocks_x - 1

    @property
    def multi_line_spacing(self) -> float:
        """Calculates the row spacing based on the offset direction

        Returns:
            float: Value for spacing each row
        """
        match self.spacing_params.offset_direction:
            case "x":
                return self.spacing_params.spacing[1] + self.block_params.width
            case "y":
                return self.spacing_params.spacing[0] + self.block_params.length

    @property
    def perpendicular_direction(self) -> OffsetDirection:
        """Defines the perpendicular direction to the offset direction

        Returns:
            OffsetDirection: Perpendicular direction
        """
        return "x" if self.spacing_params.offset_direction == "y" else "y"

    @classmethod
    def from_file(cls, file_path: pathlib.Path):
        if file_path.exists():
            yaml_vals = read_yaml(file_path)
            params = cls(**yaml_vals)
            return params
        else:
            raise Exception(f"Unable to read yaml. Filetitle {file_path.name} does not exists")
