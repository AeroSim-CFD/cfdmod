import pathlib

from pydantic import BaseModel, Field, ValidationError

from cfdmod.utils import read_yaml

__all__ = [
    "LoftCaseConfig",
]


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
    wind_source_angle: float = Field(
        ...,
        title="Wind source angle",
        description="Angle for the wind source direction."
        + "Rotated around +z axis, from the reference direction.",
    )
    upwind_elevation: float = Field(
        ...,
        title="Upwind elevation",
        description="Elevation for upwind direction.",
    )
    filter_radius: float = Field(
        0,
        title="Hole filter radius",
        description="Radius to filter out internal holes.",
    )


class LoftCaseConfig(BaseModel):
    reference_direction: tuple[float, float, float] = Field(
        [-1, 0, 0], title="Reference direction", description="Reference direction for 0° angle"
    )
    cases: dict[str, LoftParams] = Field(
        ...,
        title="Loft cases",
        description="Setup for multiple loft configurations, for each wind source direction.",
    )

    @classmethod
    def from_file(cls, file_path: pathlib.Path):
        if file_path.exists():
            yaml_vals = read_yaml(file_path)
            for case_lbl, case_dict in yaml_vals["cases"].items():
                try:
                    _ = LoftParams(**case_dict)
                except ValidationError:
                    try:
                        yaml_vals["cases"][case_lbl] = yaml_vals["cases"]["default"] | case_dict
                    except KeyError as ex:
                        raise KeyError(
                            f"Case {case_lbl} is missing fields, default is not set"
                        ) from ex
            params = cls(**yaml_vals)
            return params
        else:
            raise Exception(f"Unable to read yaml. Filetitle {file_path.name} does not exists")
