import pathlib
from typing import Annotated, Literal

from pydantic import BaseModel, Field

from cfdmod.utils import read_yaml

__all__ = [
    "OffsetDirection",
    "GenerationParams",
    "ElementParams",
    "SpacingParams",
]


OffsetDirection = Annotated[
    Literal["x", "y"], Field(description="""Define the offset direction for element lines""")
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


class ElementParams(BaseModel):
    height: float = Field(
        ...,
        title="Element height",
        description="Size of the generated elements in Z axis",
        gt=0,
    )
    width: float = Field(
        ...,
        title="Element width",
        description="Size of the generated elements in Y axis",
        gt=0,
    )


class GenerationParams(BaseModel):
    N_elements_x: int = Field(
        ...,
        title="Number of elements in X",
        description="Defines the number of elements in the X axis",
        gt=0,
    )
    N_elements_y: int = Field(
        ...,
        title="Number of elements in Y",
        description="Defines the number of elements in the Y axis",
        gt=0,
    )
    element_params: ElementParams = Field(
        ..., title="Element parameters", description="Object with element geometry parameters"
    )
    spacing_params: SpacingParams = Field(
        ..., title="Spacing parameters", description="Object with spacing parameters"
    )

    @property
    def single_line_elements(self) -> int:
        """Calculates the number of elements in a single line based on the offset direction

        Returns:
            int: Number of repetitions applied to an element to form a row
        """
        match self.spacing_params.offset_direction:
            case "x":
                return self.N_elements_x - 1
            case "y":
                return self.N_elements_y - 1

    @property
    def single_line_spacing(self) -> float:
        """Calculates the single line spacing based on the offset direction

        Returns:
            float: Value for spacing the elements in a single row
        """
        match self.spacing_params.offset_direction:
            case "x":
                return self.spacing_params.spacing[0]
            case "y":
                return self.spacing_params.spacing[1] + self.element_params.width

    @property
    def multi_line_elements(self) -> int:
        """Calculates the number of rows to be replicated based on the offset direction

        Returns:
            int: Number of repetitions applied to a row of elements
        """
        match self.spacing_params.offset_direction:
            case "x":
                return self.N_elements_y - 1
            case "y":
                return self.N_elements_x - 1

    @property
    def multi_line_spacing(self) -> float:
        """Calculates the row spacing based on the offset direction

        Returns:
            float: Value for spacing each row
        """
        match self.spacing_params.offset_direction:
            case "x":
                return self.spacing_params.spacing[1] + self.element_params.width
            case "y":
                return self.spacing_params.spacing[0]

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
            raise Exception(f"Unable to read yaml. File {file_path.name} does not exists")
