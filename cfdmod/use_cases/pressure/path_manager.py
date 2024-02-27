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
    def get_excluded_surface_path(self, cfg_lbl: str, cfg_hash: str) -> pathlib.Path:
        return self.output_path / self._FOLDERNAME / cfg_lbl / cfg_hash / "excluded_surfaces.stl"

    def get_vtp_path(self, cfg_lbl: str, cfg_hash: str) -> pathlib.Path:
        return self.output_path / self._FOLDERNAME / cfg_lbl / cfg_hash / "stats.vtp"

    def get_timeseries_df_path(self, cfg_lbl: str, cfg_hash: str) -> pathlib.Path:
        return self.output_path / self._FOLDERNAME / cfg_lbl / cfg_hash / "time_series.h5"

    def get_stats_df_path(self, cfg_lbl: str, cfg_hash: str) -> pathlib.Path:
        return self.output_path / self._FOLDERNAME / cfg_lbl / cfg_hash / "stats_df.h5"

    def get_regions_df_path(self, cfg_lbl: str, cfg_hash: str) -> pathlib.Path:
        return self.output_path / self._FOLDERNAME / cfg_lbl / cfg_hash / "regions.h5"


class CmPathManager(PathManagerBody):
    _FOLDERNAME: ClassVar[str] = "Cm"


class CfPathManager(PathManagerBody):
    _FOLDERNAME: ClassVar[str] = "Cf"


class CePathManager(PathManagerBody):
    _FOLDERNAME: ClassVar[str] = "Ce"

    def get_surface_path(self, cfg_lbl: str, cfg_hash: str, sfc_lbl: str) -> pathlib.Path:
        return self.output_path / self._FOLDERNAME / cfg_lbl / cfg_hash / f"{sfc_lbl}.regions.stl"


class CpPathManager(PathManagerBase):
    _FOLDERNAME: ClassVar[str] = "cp"

    def get_cp_stats_path(self, cfg_lbl: str, cfg_hash: str) -> pathlib.Path:
        return self.output_path / self._FOLDERNAME / cfg_lbl / cfg_hash / "stats_df.h5"

    def get_cp_t_path(self, cfg_lbl: str, cfg_hash: str) -> pathlib.Path:
        return self.output_path / self._FOLDERNAME / cfg_lbl / cfg_hash / "time_series.h5"

    def get_grouped_cp_path(self, cfg_lbl: str, cfg_hash: str) -> pathlib.Path:
        return self.output_path / self._FOLDERNAME / cfg_lbl / cfg_hash / "time_series.grouped.h5"

    def get_vtp_path(self, cfg_lbl: str, cfg_hash: str) -> pathlib.Path:
        return self.output_path / self._FOLDERNAME / cfg_lbl / cfg_hash / "stats.vtp"


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
    shutil.copy(mesh_path, path_manager.output_path / "input_cp" / mesh_path.name)
    shutil.copy(
        static_data_path, path_manager.output_path / "input_cp" / "data" / static_data_path.name
    )
    shutil.copy(
        body_data_path, path_manager.output_path / "input_cp" / "data" / body_data_path.name
    )
