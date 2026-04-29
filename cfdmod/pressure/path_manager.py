"""Path manager classes for the pressure module output directory layout."""

import pathlib
from typing import ClassVar

from pydantic import BaseModel, Field


class PathManagerBase(BaseModel):
    _FOLDERNAME: ClassVar[str]

    output_path: pathlib.Path = Field(
        ..., title="Output path", description="Path for saving output files"
    )

    def get_timeseries_path(self, cfg_lbl: str) -> pathlib.Path:
        return (
            self.output_path
            / self._FOLDERNAME
            / cfg_lbl
            / f"{self._FOLDERNAME}.time_series.h5"
        )

    def get_body_timeseries_path(self, cfg_lbl: str, body_name: str) -> pathlib.Path:
        """Per-body timeseries H5 (used by Cf/Cm where each body has its own mesh)."""
        return (
            self.output_path
            / self._FOLDERNAME
            / cfg_lbl
            / f"{body_name}.time_series.h5"
        )

    def get_config_path(self, cfg_lbl: str) -> pathlib.Path:
        return self.output_path / self._FOLDERNAME / cfg_lbl / f"{self._FOLDERNAME}.config.yaml"

    def get_results_h5_path(self) -> pathlib.Path:
        """Return path for the combined stats H5 (results.h5) shared across all coefficients."""
        return self.output_path / "results.h5"

    def get_results_xdmf_path(self) -> pathlib.Path:
        """Return path for the combined stats XDMF (results.xdmf) shared across all coefficients."""
        return self.output_path / "results.xdmf"


class CmPathManager(PathManagerBase):
    _FOLDERNAME: ClassVar[str] = "Cm"


class CfPathManager(PathManagerBase):
    _FOLDERNAME: ClassVar[str] = "Cf"


class CePathManager(PathManagerBase):
    _FOLDERNAME: ClassVar[str] = "Ce"

    def get_regions_stl_path(self, cfg_lbl: str) -> pathlib.Path:
        """STL file containing the sliced regions mesh used for Ce."""
        return self.output_path / self._FOLDERNAME / cfg_lbl / "regions.stl"


class CpPathManager(PathManagerBase):
    _FOLDERNAME: ClassVar[str] = "cp"
