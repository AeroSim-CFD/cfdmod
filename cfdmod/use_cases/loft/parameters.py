__all__ = [
    "LoftParams",
]

import pathlib

from pydantic import BaseModel, Field

from cfdmod.utils import read_yaml


class LoftParams(BaseModel):
    loft_length: float = Field(
        ...,
        title="Loft length",
        description="Minimal length of the loft.",
    )
    mesh_element_size: float = Field(
        ...,
        title="Mesh element size",
        description="Target of the output mesh element size.",
    )
    wind_source_direction: tuple[float, float, float] = Field(
        ...,
        title="Wind source direction",
        description="Direction for the wind source direction."
        + "If it flows in the positive x axis, then the source direction is -x.",
    )
    upwind_elevation: float = Field(
        ...,
        title="Upwind elevation",
        description="Elevation for upwind direction.",
    )

    @classmethod
    def from_file(cls, file_path: pathlib.Path):
        if file_path.exists():
            yaml_vals = read_yaml(file_path)
            params = cls(**yaml_vals)
            return params
        else:
            raise Exception(f"Unable to read yaml. Filetitle {file_path.name} does not exists")
