import pathlib


class CpPathManager:
    def __init__(
        self,
        output_path: str,
        config_path: str,
        mesh_path: str,
        body_data_path: str,
        static_data_path: str,
    ):
        self.output_path = pathlib.Path(output_path)
        self.config_path = pathlib.Path(config_path)
        self.mesh_path = pathlib.Path(mesh_path)
        self.body_data_path = pathlib.Path(body_data_path)
        self.static_data_path = pathlib.Path(static_data_path)

    @property
    def cp_stats_path(self):
        return self.output_path / "cp_stats.hdf"

    @property
    def cp_t_path(self):
        return self.output_path / "cp_t.hdf"

    @property
    def vtp_path(self):
        return self.output_path / "cp_stats.vtp"
