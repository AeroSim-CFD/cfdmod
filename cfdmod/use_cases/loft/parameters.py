import pathlib

from pydantic import Field, ValidationError

from cfdmod.api.configs.hashable import HashableConfig
from cfdmod.utils import read_yaml

__all__ = [
    "LoftCaseConfig",
    "LoftParams",
]


class LoftParams(HashableConfig):
    loft_radius: float = Field(
        ...,
        title="Loft radius",
        description="Radius of the circular loft projection from mesh center.",
    )
    mesh_element_size: float = Field(
        ...,
        title="Mesh element size",
        description="Target of the output mesh element size.",
    )
    upwind_elevation: float = Field(
        ...,
        title="Loft elevation",
        description="Target Z elevation for the loft base.",
    )


class LoftCaseConfig(HashableConfig):
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
