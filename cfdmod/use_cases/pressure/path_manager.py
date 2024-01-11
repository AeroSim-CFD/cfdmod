import pathlib
import shutil
from typing import ClassVar

from pydantic import BaseModel, Field

from cfdmod.utils import create_folder_path


class PathManagerBase(BaseModel):
    _FOLDERNAME: ClassVar[str]

    output_path: pathlib.Path = Field(
        ..., title="Output path", description="Path for saving output files"
    )


class PathManagerBody(PathManagerBase):
    def get_excluded_surface_path(self, cfg_label: str) -> pathlib.Path:
        return (
            self.output_path / cfg_label / self._FOLDERNAME / "surfaces" / "excluded_surfaces.stl"
        )

    def get_vtp_path(self, body_label: str, cfg_label: str) -> pathlib.Path:
        return self.output_path / cfg_label / self._FOLDERNAME / f"{body_label}.stats.vtp"

    def get_timeseries_df_path(self, body_label: str, cfg_label: str) -> pathlib.Path:
        return (
            self.output_path
            / cfg_label
            / self._FOLDERNAME
            / "time_series"
            / f"{body_label}.time_series.h5"
        )

    def get_stats_df_path(self, body_label: str, cfg_label: str) -> pathlib.Path:
        return self.output_path / cfg_label / self._FOLDERNAME / "stats" / f"{body_label}.stats.h5"


class CmPathManager(PathManagerBody):
    _FOLDERNAME: ClassVar[str] = "Cm"


class CfPathManager(PathManagerBody):
    _FOLDERNAME: ClassVar[str] = "Cf"


class CePathManager(PathManagerBody):
    _FOLDERNAME: ClassVar[str] = "Ce"

    def get_surface_path(self, sfc_label: str, cfg_label: str) -> pathlib.Path:
        return (
            self.output_path
            / cfg_label
            / self._FOLDERNAME
            / "surfaces"
            / f"{sfc_label}.regions.stl"
        )

    def get_regions_df_path(self, sfc_label: str, cfg_label: str) -> pathlib.Path:
        return (
            self.output_path / cfg_label / self._FOLDERNAME / "regions" / f"regions.{sfc_label}.h5"
        )


class CpPathManager(PathManagerBase):
    _FOLDERNAME: ClassVar[str] = "cp"

    def get_cp_stats_path(self, cfg_label: str) -> pathlib.Path:
        return self.output_path / cfg_label / self._FOLDERNAME / "cp_stats.h5"

    def get_cp_t_path(self, cfg_label: str) -> pathlib.Path:
        return self.output_path / cfg_label / self._FOLDERNAME / "cp_t.h5"

    def get_vtp_path(self, cfg_label: str) -> pathlib.Path:
        return self.output_path / cfg_label / self._FOLDERNAME / "cp_stats.vtp"


def copy_input_artifacts(
    cfg_path: pathlib.Path,
    mesh_path: pathlib.Path,
    static_data_path: pathlib.Path,
    body_data_path: pathlib.Path,
    path_manager: CpPathManager,
):
    create_folder_path(path_manager.output_path / "input_cp")
    create_folder_path(path_manager.output_path / "input_cp" / "data")

    shutil.copy(cfg_path, path_manager.output_path / "input_cp" / cfg_path.name)
    mesh_output = path_manager.output_path / "input_cp" / mesh_path.parent.name
    if mesh_output.is_dir():
        # Overwrite the files in the output folder
        for file in mesh_path.parent.iterdir():
            shutil.copy(
                file,
                mesh_output / file.name,
            )
    else:
        shutil.copytree(mesh_path.parent, mesh_output)
    shutil.copy(
        static_data_path, path_manager.output_path / "input_cp" / "data" / static_data_path.name
    )
    shutil.copy(
        body_data_path, path_manager.output_path / "input_cp" / "data" / body_data_path.name
    )
