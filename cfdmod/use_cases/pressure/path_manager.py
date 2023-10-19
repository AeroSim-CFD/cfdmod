import pathlib
import shutil

from pydantic import BaseModel, Field

from cfdmod.utils import create_folder_path


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
    shutil.copytree(
        mesh_path.parent, path_manager.output_path / "input_cp" / mesh_path.parent.name
    )
    shutil.copy(
        static_data_path, path_manager.output_path / "input_cp" / "data" / static_data_path.name
    )
    shutil.copy(
        body_data_path, path_manager.output_path / "input_cp" / "data" / body_data_path.name
    )
