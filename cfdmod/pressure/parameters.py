"""Pressure module Pydantic configuration models.

All configuration models for Cp, Cf, Cm, Ce and supporting types.
"""

from __future__ import annotations

__all__ = [
    # Base
    "BasePressureConfig",
    # Statistics
    "Statistics",
    "ExtremeMethods",
    "ExtremeAbsoluteParamsModel",
    "ExtremeGumbelParamsModel",
    "ExtremePeakParamsModel",
    "ExtremeMovingAverageParamsModel",
    "MeanEquivalentParamsModel",
    "StatisticsParamsModel",
    "BasicStatisticModel",
    "ParameterizedStatisticModel",
    # Zoning / body config
    "ZoningModel",
    "BodyDefinition",
    "BodyConfig",
    "MomentBodyConfig",
    # Cp
    "CpConfig",
    "CpCaseConfig",
    # Cf
    "CfConfig",
    "CfCaseConfig",
    # Cm
    "CmConfig",
    "CmCaseConfig",
    # Ce
    "ZoningBuilder",
    "ExceptionZoningModel",
    "ZoningConfig",
    "CeConfig",
    "CeCaseConfig",
    # Directions
    "AxisDirections",
]

import itertools
import math
import pathlib
from typing import Annotated, Literal, get_args

import numpy as np
from pydantic import BaseModel, Field, field_validator, model_validator

from cfdmod.config.hashable import HashableConfig
from cfdmod.io.geometry.transformation_config import TransformationConfig
from cfdmod.utils import read_yaml

# ---------------------------------------------------------------------------
# Statistics models
# ---------------------------------------------------------------------------

Statistics = Literal["max", "min", "rms", "mean", "mean_eq", "skewness", "kurtosis"]
ExtremeMethods = Literal["Gumbel", "Peak", "Absolute", "Moving Average"]
AxisDirections = Literal["x", "y", "z"]


class ExtremeAbsoluteParamsModel(BaseModel):
    method_type: Literal["Absolute"] = "Absolute"


class ExtremeGumbelParamsModel(BaseModel):
    method_type: Literal["Gumbel"] = "Gumbel"
    peak_duration: float
    event_duration: float
    n_subdivisions: int = 10
    non_exceedance_probability: float = Field(0.78, gt=0, lt=1)
    full_scale_U_H: float = Field(gt=0)
    full_scale_characteristic_length: float = Field(gt=0)

    @property
    def yR(self):
        if not hasattr(self, "_yR"):
            self._yR = -np.log(-np.log(self.non_exceedance_probability))
        return self._yR


class ExtremePeakParamsModel(BaseModel):
    method_type: Literal["Peak"] = "Peak"
    peak_factor: float


class ExtremeMovingAverageParamsModel(BaseModel):
    method_type: Literal["Moving Average"] = "Moving Average"
    window_size_interval: float = Field(gt=0)
    full_scale_U_H: float = Field(gt=0)
    full_scale_characteristic_length: float = Field(gt=0)


class MeanEquivalentParamsModel(BaseModel):
    scale_factor: float = Field(default=0.61, gt=0, le=1)


StatisticsParamsModel = (
    MeanEquivalentParamsModel
    | ExtremeGumbelParamsModel
    | ExtremePeakParamsModel
    | ExtremeAbsoluteParamsModel
    | ExtremeMovingAverageParamsModel
)


class BasicStatisticModel(BaseModel):
    stats: Statistics
    display_name: str = ""


class ParameterizedStatisticModel(BasicStatisticModel):
    params: StatisticsParamsModel

    @field_validator("params", mode="before")
    def validate_params(cls, v):
        if not isinstance(v, dict):
            return v
        if "method_type" in v:
            if v["method_type"] == "Gumbel":
                return ExtremeGumbelParamsModel(**v)
            elif v["method_type"] == "Peak":
                return ExtremePeakParamsModel(**v)
            elif v["method_type"] == "Absolute":
                return ExtremeAbsoluteParamsModel(**v)
            elif v["method_type"] == "Moving Average":
                return ExtremeMovingAverageParamsModel(**v)
            else:
                available = get_args(ExtremeMethods)
                raise ValueError(
                    f"Unknown method {v['method_type']}, available: {available}"
                )
        return MeanEquivalentParamsModel(**v)


# ---------------------------------------------------------------------------
# Base pressure config
# ---------------------------------------------------------------------------


class BasePressureConfig(BaseModel):
    statistics: list[BasicStatisticModel | ParameterizedStatisticModel] = Field(
        ...,
        title="List of statistics",
        description="List of statistics to calculate from pressure coefficient signal",
    )

    @field_validator("statistics", mode="before")
    def validate_statistics(cls, v):
        if isinstance(v[0], dict):
            stats_types = [s["stats"] for s in v]
        else:
            stats_types = [s.stats for s in v]
        if len(set(stats_types)) != len(stats_types):
            raise Exception("Duplicated statistics! Only one of each type allowed")
        if "mean_eq" in stats_types:
            if any(s not in stats_types for s in ["mean", "min", "max"]):
                raise Exception("mean_eq requires mean, min and max statistics")
        validated_list = []
        for statistic in v:
            if isinstance(statistic, dict):
                if "params" in statistic:
                    validated_list.append(ParameterizedStatisticModel(**statistic))
                else:
                    validated_list.append(BasicStatisticModel(**statistic))
            else:
                validated_list.append(statistic)
        return validated_list


# ---------------------------------------------------------------------------
# Zoning / body config
# ---------------------------------------------------------------------------


class ZoningModel(HashableConfig):
    x_intervals: list[float] = Field(
        [float("-inf"), float("inf")],
        title="X intervals list",
        description="Values for the X axis intervals list, it must be unique",
    )
    y_intervals: list[float] = Field(
        [float("-inf"), float("inf")],
        title="Y intervals list",
        description="Values for the Y axis intervals list, it must be unique",
    )
    z_intervals: list[float] = Field(
        [float("-inf"), float("inf")],
        title="Z intervals list",
        description="Values for the Z axis intervals list, it must be unique",
    )

    @field_validator("x_intervals", "y_intervals", "z_intervals")
    def validate_interval(cls, v):
        if len(v) != len(set(v)):
            raise Exception("Invalid region intervals, values must not repeat")
        return v

    @field_validator("x_intervals", "y_intervals", "z_intervals")
    def validate_intervals(cls, v):
        if len(v) == 0:
            v = [float("-inf"), float("inf")]
        elif len(v) < 2:
            raise Exception("Interval must have at least 2 values")
        for i in range(len(v) - 1):
            if v[i] >= v[i + 1]:
                raise Exception("Interval must have ascending order")
        return v

    def ignore_axis(self, axis: int) -> ZoningModel:
        new_zoning = self.model_copy()
        if axis == 0:
            new_zoning.x_intervals = [float("-inf"), float("inf")]
        elif axis == 1:
            new_zoning.y_intervals = [float("-inf"), float("inf")]
        elif axis == 2:
            new_zoning.z_intervals = [float("-inf"), float("inf")]
        return new_zoning

    def offset_limits(self, offset_value: float) -> ZoningModel:
        offsetted = self.model_copy()
        x_int = self.x_intervals[:]
        y_int = self.y_intervals[:]
        z_int = self.z_intervals[:]
        x_int[0] -= offset_value
        x_int[-1] += offset_value
        y_int[0] -= offset_value
        y_int[-1] += offset_value
        z_int[0] -= offset_value
        z_int[-1] += offset_value
        offsetted.x_intervals = x_int[:]
        offsetted.y_intervals = y_int[:]
        offsetted.z_intervals = z_int[:]
        return offsetted

    def get_regions(self) -> list[tuple[tuple[float, float], ...]]:
        def _build_intervals(intervals: list[float]):
            return [(intervals[i], intervals[i + 1]) for i in range(len(intervals) - 1)]

        x_regions = _build_intervals(self.x_intervals)
        y_regions = _build_intervals(self.y_intervals)
        z_regions = _build_intervals(self.z_intervals)
        return list(itertools.product(x_regions, y_regions, z_regions))

    def get_regions_df(self):
        import pandas as pd

        regions = self.get_regions()
        regions_dct: dict = {
            "x_min": [],
            "x_max": [],
            "y_min": [],
            "y_max": [],
            "z_min": [],
            "z_max": [],
        }
        for region in regions:
            for i, d in enumerate(["x", "y", "z"]):
                regions_dct[f"{d}_min"].append(region[i][0])
                regions_dct[f"{d}_max"].append(region[i][1])
        df = pd.DataFrame(regions_dct)
        df["region_idx"] = df.index
        return df


class BodyDefinition(HashableConfig):
    surfaces: list[str] = Field(
        ..., title="Body's surfaces", description="List of surfaces that compose the body"
    )

    @field_validator("surfaces")
    def validate_surface_list(cls, v):
        if len(v) != len(set(v)):
            raise Exception("Invalid surface list, names must not repeat")
        return v


class BodyConfig(HashableConfig):
    name: str = Field(
        ...,
        title="Body's name",
        description="Name of the body defined in the bodies section",
    )
    sub_bodies: ZoningModel = Field(
        ZoningModel(
            x_intervals=[float("-inf"), float("inf")],
            y_intervals=[float("-inf"), float("inf")],
            z_intervals=[float("-inf"), float("inf")],
        ),
        title="Sub body intervals",
        description="Intervals that section the body into sub-bodies",
    )


class MomentBodyConfig(BodyConfig):
    lever_origin: tuple[float, float, float] = Field(
        ...,
        title="Lever origin",
        description="Reference point to evaluate the lever for moment calculations",
    )


# ---------------------------------------------------------------------------
# Ce shape coefficient zoning config
# ---------------------------------------------------------------------------


class ExceptionZoningModel(ZoningModel):
    surfaces: list[str] = Field(
        ...,
        title="List of surfaces",
        description="Surfaces to include in the exceptional zoning",
    )

    @field_validator("surfaces")
    def validate_surface_list(cls, v):
        if len(v) != len(set(v)):
            raise Exception("Invalid exceptions surface list, names must not repeat")
        if len(v) == 0:
            raise Exception("Invalid exceptions surface list, list must not be empty")
        return v


class ZoningConfig(BaseModel):
    global_zoning: ZoningModel = Field(
        ...,
        title="Global Zoning Config",
        description="Default Zoning Config applied to all surfaces",
    )
    no_zoning: list[str] = Field(
        [],
        title="No Zoning Surfaces",
        description="List of surfaces to ignore region mesh generation",
    )
    exclude: list[str] = Field(
        [],
        title="Surfaces to exclude",
        description="Surfaces to ignore when calculating shape coefficient",
    )
    exceptions: dict[str, ExceptionZoningModel] = Field(
        {},
        title="Dict with specific zoning",
        description="Specific zoning config that overrides the global zoning",
    )

    @model_validator(mode="after")
    def validate_config(self) -> ZoningConfig:
        common_surfaces = set(self.no_zoning).intersection(
            set(self.exclude), set(self.surfaces_in_exception)
        )
        if len(common_surfaces) != 0:
            raise Exception("Surface name must not appear in two different zoning rules")
        return self

    @field_validator("exceptions")
    def validate_exceptions(cls, v):
        exceptions = []
        for exception_cfg in v.values():
            exceptions += exception_cfg.surfaces
        if len(exceptions) != len(set(exceptions)):
            raise Exception("Invalid exceptions list, surface names must not repeat")
        return v

    @field_validator("no_zoning", "exclude")
    def validate_surface_list(cls, v):
        if len(v) != len(set(v)):
            raise Exception("Invalid surface list, names must not repeat")
        return v

    @property
    def surfaces_in_exception(self) -> list[str]:
        exceptions = []
        for exception_cfg in self.exceptions.values():
            exceptions += exception_cfg.surfaces
        return exceptions

    @property
    def surfaces_listed(self) -> list[str]:
        return self.surfaces_in_exception + self.no_zoning + self.exclude

    @classmethod
    def from_file(cls, filename: pathlib.Path) -> ZoningConfig:
        yaml_vals = read_yaml(filename)
        return cls.model_validate(yaml_vals)


# ---------------------------------------------------------------------------
# Cp
# ---------------------------------------------------------------------------


class CpConfig(HashableConfig, BasePressureConfig):
    """Configuration to calculate pressure coefficient."""

    timestep_range: tuple[float, float] = Field(
        ...,
        title="Timestep Range",
        description="Interval between start and end steps to slice data",
    )
    macroscopic_type: Annotated[
        Literal["rho", "pressure"],
        Field(
            "rho",
            title="Macroscopic type",
            description="Macroscopic type in files, LBM density (rho) or pressure",
        ),
    ]
    fluid_density: Annotated[
        float,
        Field(
            1,
            title="Fluid density",
            description="Fluid density for Cp calculation",
        ),
    ]
    simul_U_H: float = Field(
        ...,
        title="Simulation Flow Velocity",
        description="Simulation flow velocity to calculate dynamic pressure and time scales",
    )
    simul_characteristic_length: float = Field(
        ...,
        title="Simulation Characteristic Length",
        description="Simulation characteristic length to convert time scales",
    )
    time_scale_multiplier: float = Field(
        1,
        title="Atmospheric integral length scale corrector",
        description="Multiplier to correct the effective integral length scale",
    )


class CpCaseConfig(HashableConfig):
    pressure_coefficient: dict[str, CpConfig] = Field(
        ...,
        title="Pressure Coefficient configs",
        description="Dictionary with Pressure Coefficient configuration",
    )

    @classmethod
    def from_file(cls, filename: pathlib.Path) -> CpCaseConfig:
        yaml_vals = read_yaml(filename)
        return cls(**yaml_vals)


# ---------------------------------------------------------------------------
# Cf
# ---------------------------------------------------------------------------


class CfConfig(HashableConfig, BasePressureConfig):
    """Configuration for force coefficient."""

    bodies: list[BodyConfig] = Field(
        ...,
        title="Bodies configuration",
        description="Bodies to process and their zoning config",
    )
    nominal_area: float = Field(
        0,
        title="Nominal Area",
        description="Nominal area for force coefficient calculation",
    )
    directions: list[AxisDirections] = Field(
        ...,
        title="List of directions",
        description="Directions for which force coefficient will be calculated",
    )
    transformation: TransformationConfig = Field(
        TransformationConfig(),
        title="Transformation config",
        description="Configuration for mesh transformation",
    )


class CfCaseConfig(HashableConfig):
    bodies: dict[str, BodyDefinition] = Field(
        ..., title="Bodies definition", description="Named bodies definition"
    )
    force_coefficient: dict[str, CfConfig] = Field(
        ...,
        title="Force Coefficient configs",
        description="Dictionary with Force Coefficient configuration",
    )

    @model_validator(mode="after")
    def validate_body_list(self):
        for body_label in [b.name for cfg in self.force_coefficient.values() for b in cfg.bodies]:
            if body_label not in self.bodies:
                raise Exception(f"Body {body_label} is not defined in the configuration file")
        return self

    @model_validator(mode="after")
    def validate_body_surfaces(self):
        for cfg_lbl, cfg in self.force_coefficient.items():
            all_sfc = [sfc for b in cfg.bodies for sfc in self.bodies[b.name].surfaces]
            if len(all_sfc) != len(set(all_sfc)):
                raise Exception(f"Config {cfg_lbl} repeats surface in more than one body.")
        return self

    @classmethod
    def from_file(cls, filename: pathlib.Path) -> CfCaseConfig:
        yaml_vals = read_yaml(filename)
        return cls(**yaml_vals)


# ---------------------------------------------------------------------------
# Cm
# ---------------------------------------------------------------------------


class CmConfig(HashableConfig, BasePressureConfig):
    """Configuration for moment coefficient."""

    bodies: list[MomentBodyConfig] = Field(
        ...,
        title="Bodies configuration",
        description="Bodies to process and their zoning config",
    )
    nominal_volume: float = Field(
        0,
        title="Nominal Volume",
        description="Nominal volume for moment coefficient. If zero, uses tribute volume",
    )
    directions: list[AxisDirections] = Field(
        ...,
        title="List of directions",
        description="Directions for which moment coefficient will be calculated",
    )
    transformation: TransformationConfig = Field(
        TransformationConfig(),
        title="Transformation config",
        description="Configuration for mesh transformation",
    )


class CmCaseConfig(HashableConfig):
    bodies: dict[str, BodyDefinition] = Field(
        ..., title="Bodies definition", description="Named bodies definition"
    )
    moment_coefficient: dict[str, CmConfig] = Field(
        ...,
        title="Moment Coefficient configs",
        description="Dictionary with Moment Coefficient configuration",
    )

    @model_validator(mode="after")
    def validate_body_list(self):
        for body_label in [
            b.name for cfg in self.moment_coefficient.values() for b in cfg.bodies
        ]:
            if body_label not in self.bodies:
                raise Exception(f"Body {body_label} is not defined in the configuration file")
        return self

    @model_validator(mode="after")
    def validate_body_surfaces(self):
        for cfg_lbl, cfg in self.moment_coefficient.items():
            all_sfc = [sfc for b in cfg.bodies for sfc in self.bodies[b.name].surfaces]
            if len(all_sfc) != len(set(all_sfc)):
                raise Exception(f"Config {cfg_lbl} repeats surface in more than one body.")
        return self

    @classmethod
    def from_file(cls, filename: pathlib.Path) -> CmCaseConfig:
        yaml_vals = read_yaml(filename)
        return cls(**yaml_vals)


# ---------------------------------------------------------------------------
# Ce (shape coefficient)
# ---------------------------------------------------------------------------


class ZoningBuilder(HashableConfig):
    """Reference to an external YAML file containing ZoningConfig."""

    yaml: str = Field(
        ...,
        title="Path to Zoning yaml",
        description="Path to Zoning yaml for constructing zoning configuration",
    )
    _base_path: pathlib.Path = pathlib.Path("./")

    def to_zoning_config(self) -> ZoningConfig:
        return ZoningConfig.from_file(self._base_path / pathlib.Path(self.yaml))


class CeConfig(HashableConfig, BasePressureConfig):
    """Configuration for shape coefficient."""

    zoning: ZoningConfig | ZoningBuilder = Field(
        ...,
        title="Zoning configuration",
        description="Zoning configuration with intervals information",
    )
    sets: dict[str, list[str]] = Field(
        {},
        title="Surface sets",
        description="Combine multiple surfaces into a set",
    )
    transformation: TransformationConfig = Field(
        TransformationConfig(),
        title="Transformation config",
        description="Configuration for mesh transformation",
    )

    @property
    def surfaces_in_sets(self):
        return [sfc for sfc_list in self.sets.values() for sfc in sfc_list]

    @field_validator("sets")
    def validate_sets(cls, v):
        surface_list = [sfc for sfc_list in v.values() for sfc in sfc_list]
        if len(surface_list) != len(set(surface_list)):
            raise Exception("A surface cannot be listed in more than one set")
        return v

    def to_zoning(self):
        if isinstance(self.zoning, ZoningBuilder):
            self.zoning = self.zoning.to_zoning_config()

    def validate_zoning_surfaces(self):
        common = set(self.surfaces_in_sets).intersection(
            set(self.zoning.surfaces_listed)  # type: ignore
        )
        if len(common) != 0:
            raise Exception("Surfaces inside a set cannot be listed in zoning")


class CeCaseConfig(HashableConfig):
    shape_coefficient: dict[str, CeConfig] = Field(
        ...,
        title="Shape Coefficient configs",
        description="Dictionary of shape coefficient configurations",
    )

    @classmethod
    def from_file(cls, filename: pathlib.Path) -> CeCaseConfig:
        yaml_vals = read_yaml(filename)
        cfg = cls(**yaml_vals)
        for s in cfg.shape_coefficient.values():
            s.zoning._base_path = filename.parent  # type: ignore
            s.to_zoning()
            s.validate_zoning_surfaces()
        return cfg
