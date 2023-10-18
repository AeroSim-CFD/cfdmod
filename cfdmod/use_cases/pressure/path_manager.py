import pathlib

from pydantic import BaseModel, Field, validator


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
