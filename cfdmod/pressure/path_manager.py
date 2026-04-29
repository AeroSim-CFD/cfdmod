"""Path manager classes for the pressure module output layout.

Flat by default: every artifact for a given (coefficient, cfg_lbl[, body])
sits directly in ``output_path``. File names embed the discriminators with
dots, e.g.::

    output/cp.default.time_series.h5
    output/cp.default.time_series.xdmf
    output/cp.default.config.yaml
    output/Cf.containers.pack.time_series.h5
    output/Cm.containers.pack.time_series.h5
    output/Ce.measurement_1.time_series.h5
    output/Ce.measurement_1.regions.stl
    output/results.h5
    output/results.xdmf

This keeps "open the output folder, see everything" workflows trivial. If
nested layouts are ever needed, override the ``get_*`` methods on a subclass.
"""

import pathlib
from typing import ClassVar

from pydantic import BaseModel, Field


class PathManagerBase(BaseModel):
    _PREFIX: ClassVar[str]  # short label for filename prefix (e.g. "cp", "Cf")

    output_path: pathlib.Path = Field(
        ..., title="Output path", description="Path for saving output files"
    )

    def _stem(self, *parts: str) -> str:
        return ".".join((self._PREFIX, *parts))

    def get_timeseries_path(self, cfg_lbl: str) -> pathlib.Path:
        return self.output_path / f"{self._stem(cfg_lbl)}.time_series.h5"

    def get_body_timeseries_path(self, cfg_lbl: str, body_name: str) -> pathlib.Path:
        """Per-body timeseries H5 (used by Cf/Cm where each body has its own mesh)."""
        return self.output_path / f"{self._stem(cfg_lbl, body_name)}.time_series.h5"

    def get_results_h5_path(self) -> pathlib.Path:
        """Return path for the combined stats H5 (shared across all coefficients)."""
        return self.output_path / "results.h5"

    def get_results_xdmf_path(self) -> pathlib.Path:
        """Return path for the combined stats XDMF (shared across all coefficients)."""
        return self.output_path / "results.xdmf"


class CmPathManager(PathManagerBase):
    _PREFIX: ClassVar[str] = "Cm"


class CfPathManager(PathManagerBase):
    _PREFIX: ClassVar[str] = "Cf"


class CePathManager(PathManagerBase):
    _PREFIX: ClassVar[str] = "Ce"

    def get_regions_stl_path(self, cfg_lbl: str) -> pathlib.Path:
        """STL file containing the sliced regions mesh used for Ce."""
        return self.output_path / f"{self._stem(cfg_lbl)}.regions.stl"


class CpPathManager(PathManagerBase):
    _PREFIX: ClassVar[str] = "cp"
