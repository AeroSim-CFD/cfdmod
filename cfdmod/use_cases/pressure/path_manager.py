import pathlib

from cfdmod.utils import create_folder_path


class CePathManager:
    def __init__(
        self,
        output_path: str,
        config_path: str,
        mesh_path: str,
        cp_data_path: str,
    ):
        self.output_path = pathlib.Path(output_path)
        self.config_path = pathlib.Path(config_path)
        self.mesh_path = pathlib.Path(mesh_path)
        self.cp_data_path = pathlib.Path(cp_data_path)

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
