import pathlib
import shutil

from pydantic import BaseModel, Field

from cfdmod.utils import create_folder_path


class CePathManager(BaseModel):
    output_path: pathlib.Path = Field(
        ..., title="Output path", description="Path for saving output files"
    )

    def get_surface_path(self, sfc_label: str, cfg_label: str) -> pathlib.Path:
        create_folder_path(self.output_path / cfg_label / "surfaces")
        return self.output_path / cfg_label / "surfaces" / f"{sfc_label}.regions.stl"

    def get_vtp_path(self, body_label: str, cfg_label: str) -> pathlib.Path:
        create_folder_path(self.output_path / cfg_label)
        return self.output_path / cfg_label / f"{body_label}.regions.vtp"

    def get_regions_df_path(self, sfc_label: str, cfg_label: str) -> pathlib.Path:
        create_folder_path(self.output_path / cfg_label / "regions")
        return self.output_path / cfg_label / "regions" / f"regions.{sfc_label}.hdf"

    def get_timeseries_df_path(self, sfc_label: str, cfg_label: str) -> pathlib.Path:
        create_folder_path(self.output_path / cfg_label / "time_series")
        return self.output_path / cfg_label / "time_series" / f"Ce_t.{sfc_label}.hdf"

    def get_stats_df_path(self, sfc_label: str, cfg_label: str) -> pathlib.Path:
        create_folder_path(self.output_path / cfg_label / "stats")
        return self.output_path / cfg_label / "stats" / f"Ce_stats.{sfc_label}.hdf"


class CpPathManager(BaseModel):
    output_path: pathlib.Path = Field(
        ..., title="Output path", description="Path for saving output files"
    )

    @property
    def cp_stats_path(self):
        return self.output_path / "cp_stats.hdf"

    @property
    def cp_t_path(self):
        return self.output_path / "cp_t.hdf"

    @property
    def vtp_path(self):
        return self.output_path / "cp_stats.vtp"


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
