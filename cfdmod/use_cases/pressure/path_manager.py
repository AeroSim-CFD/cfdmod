import pathlib


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
        return self.output_path / cfg_label / f"{sfc_label}.regions.stl"

    def get_vtp_path(self, body_label: str, cfg_label: str) -> pathlib.Path:
        return self.output_path / cfg_label / f"{body_label}.regions.vtp"
