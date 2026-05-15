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
import pathlib
from typing import Annotated, Literal, get_args

import numpy as np
from pydantic import BaseModel, Field, field_validator, model_validator

from cfdmod.geometry.grouping import (
    BySurfaceGrouping,
    ByZoningGrouping,
    GroupingSpec,
)
from cfdmod.io.geometry.transformation_config import TransformationConfig
from cfdmod.utils import read_yaml

# ---------------------------------------------------------------------------
# Statistics models
# ---------------------------------------------------------------------------

Statistics = Literal["max", "min", "rms", "mean", "mean_eq", "skewness", "kurtosis"]
ExtremeMethods = Literal["Gumbel", "Peak", "Absolute"]
AxisDirections = Literal["x", "y", "z"]


class ExtremeAbsoluteParamsModel(BaseModel):
    method_type: Literal["Absolute"] = "Absolute"


class ExtremeGumbelParamsModel(BaseModel):
    method_type: Literal["Gumbel"] = "Gumbel"
    peak_duration: float
    event_duration: float
    n_subdivisions: int = 10
    non_exceedance_probability: Annotated[float, Field(0.78, gt=0, lt=1)]
    # Optional in Cf/Cm: when omitted, the runner inherits from the Cp config
    # used to produce the input cp_h5 (simul_U_H / simul_characteristic_length
    # are persisted in /processing_metadata).
    full_scale_U_H: Annotated[float | None, Field(default=None, gt=0)]
    full_scale_characteristic_length: Annotated[float | None, Field(default=None, gt=0)]

    @property
    def yR(self):
        if not hasattr(self, "_yR"):
            self._yR = -np.log(-np.log(self.non_exceedance_probability))
        return self._yR


class ExtremePeakParamsModel(BaseModel):
    method_type: Literal["Peak"] = "Peak"
    peak_factor: float


class MeanEquivalentParamsModel(BaseModel):
    scale_factor: Annotated[float, Field(default=0.61, gt=0, le=1)]


StatisticsParamsModel = (
    MeanEquivalentParamsModel
    | ExtremeGumbelParamsModel
    | ExtremePeakParamsModel
    | ExtremeAbsoluteParamsModel
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
            else:
                available = get_args(ExtremeMethods)
                raise ValueError(f"Unknown method {v['method_type']}, available: {available}")
        return MeanEquivalentParamsModel(**v)


# ---------------------------------------------------------------------------
# Base pressure config
# ---------------------------------------------------------------------------


class _YamlConfig(BaseModel):
    """Mixin for top-level case configs that load from a YAML file."""

    @classmethod
    def from_file(cls, filename: pathlib.Path):
        return cls.model_validate(read_yaml(filename))


class BasePressureConfig(BaseModel):
    statistics: Annotated[
        list[BasicStatisticModel | ParameterizedStatisticModel],
        Field(
            ...,
            title="List of statistics",
            description="List of statistics to calculate from pressure coefficient signal",
        ),
    ]

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


class ZoningModel(BaseModel):
    x_intervals: Annotated[
        list[float],
        Field(
            [float("-inf"), float("inf")],
            title="X intervals list",
            description="Values for the X axis intervals list, it must be unique",
        ),
    ]
    y_intervals: Annotated[
        list[float],
        Field(
            [float("-inf"), float("inf")],
            title="Y intervals list",
            description="Values for the Y axis intervals list, it must be unique",
        ),
    ]
    z_intervals: Annotated[
        list[float],
        Field(
            [float("-inf"), float("inf")],
            title="Z intervals list",
            description="Values for the Z axis intervals list, it must be unique",
        ),
    ]

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


class BodyDefinition(BaseModel):
    surfaces: Annotated[
        list[str],
        Field(..., title="Body's surfaces", description="List of surfaces that compose the body"),
    ]

    @field_validator("surfaces")
    def validate_surface_list(cls, v):
        if len(v) != len(set(v)):
            raise Exception("Invalid surface list, names must not repeat")
        return v


class BodyConfig(BaseModel):
    name: Annotated[
        str,
        Field(
            ...,
            title="Body's name",
            description=(
                "Name of the body defined in the bodies section. Used both as a "
                "lookup key against ``bodies_definition`` and as a filename "
                "component in the flat output layout (e.g. "
                "``Cf.{cfg_lbl}.{body_name}.time_series.h5``). "
                "Avoid characters that conflict with the dot separator in the "
                "filename layout; the moment-coefficient case expansion produces "
                "derived body names of the form ``{name}.{case_label}``, so a "
                "literal '.' inside ``name`` would make those harder to parse."
            ),
        ),
    ]
    sub_bodies: Annotated[
        ZoningModel,
        Field(
            ZoningModel(
                x_intervals=[float("-inf"), float("inf")],
                y_intervals=[float("-inf"), float("inf")],
                z_intervals=[float("-inf"), float("inf")],
            ),
            title="Sub body intervals",
            description="Intervals that section the body into sub-bodies",
        ),
    ]
    groupings: Annotated[
        list[GroupingSpec] | None,
        Field(
            None,
            title="Triangle-grouping pipeline (replaces sub_bodies when set)",
            description=(
                "Optional explicit triangle-grouping chain. When set, replaces "
                "the implicit (BySurface + ByZoning(sub_bodies)) chain entirely. "
                "Use this to express groupings the canonical chain cannot, e.g. "
                "by_connectivity or arbitrary sub-set composition. Setting both "
                "groupings and a non-default sub_bodies is rejected to keep the "
                "active chain unambiguous."
            ),
        ),
    ]

    @model_validator(mode="after")
    def _check_groupings_vs_sub_bodies(self) -> "BodyConfig":
        if self.groupings is None:
            return self
        # sub_bodies has a default, so we treat the default value as
        # "not set". Any non-default zoning + an explicit groupings chain
        # is ambiguous.
        sb = self.sub_bodies
        is_default = (
            sb.x_intervals == [float("-inf"), float("inf")]
            and sb.y_intervals == [float("-inf"), float("inf")]
            and sb.z_intervals == [float("-inf"), float("inf")]
        )
        if not is_default:
            raise ValueError(
                f"BodyConfig {self.name!r}: cannot set both 'groupings' and "
                "a non-default 'sub_bodies'; the explicit chain takes precedence "
                "but mixing is ambiguous. Move the zoning into the chain."
            )
        return self

    def resolved_groupings(self, sfc_list: list[str]) -> list[GroupingSpec]:
        """Return the grouping chain to apply for this body.

        Honors an explicit ``groupings`` field when set; otherwise builds
        the canonical ``[BySurface, ByZoning(sub_bodies)]`` chain that
        reproduces the legacy region-label format.
        """
        if self.groupings is not None:
            return list(self.groupings)
        sfcs = list(sfc_list) if sfc_list else []
        return [
            BySurfaceGrouping(sets={self.name: sfcs}),
            ByZoningGrouping(
                x_intervals=list(self.sub_bodies.x_intervals),
                y_intervals=list(self.sub_bodies.y_intervals),
                z_intervals=list(self.sub_bodies.z_intervals),
                name_template="{idx}-" + self.name,
                restrict_to=[self.name],
            ),
        ]


class MomentBodyConfig(BodyConfig):
    lever_origin: Annotated[
        tuple[float, float, float],
        Field(
            (0.0, 0.0, 0.0),
            title="Lever origin (fixed-strategy fallback)",
            description=(
                "Reference point used as the moment center for every triangle when "
                "lever_strategy='fixed'. Also serves as the fallback origin for "
                "regions not covered by region_lever_origins under any strategy."
            ),
        ),
    ]
    lever_strategy: Annotated[
        Literal["fixed", "region_base", "region_bbox_corners_xy"],
        Field(
            "fixed",
            title="Lever-origin strategy",
            description=(
                "How to derive the moment center for each region in this body. "
                "'fixed' (default): every triangle uses lever_origin. "
                "'region_base': per region, derive (mean_x, mean_y, min_z) of "
                "the region's triangle vertices -- footprint centroid at the "
                "lowest z. 'region_bbox_corners_xy': expand into 4 independent "
                "cases per body (xmin_ymin, xmin_ymax, xmax_ymin, xmax_ymax), "
                "each with its own per-region origin at the corresponding xy "
                "corner of the region bbox at z=min. Useful for finding the "
                "worst-case overturning moment about a footprint edge."
            ),
        ),
    ]
    region_lever_origins: Annotated[
        dict[int, tuple[float, float, float]] | None,
        Field(
            None,
            title="Per-region lever origins (explicit overrides)",
            description=(
                "Explicit moment centers keyed by the integer region index "
                "(0, 1, ... matching the order produced by sub_bodies). Takes "
                "precedence over both lever_strategy and lever_origin. Use this "
                "for HFPI-style analyses where the center of mass of each body "
                "or container is known externally."
            ),
        ),
    ]
    lever_origin_cases: Annotated[
        dict[str, dict[int, tuple[float, float, float]]] | None,
        Field(
            None,
            title="Per-case per-region lever origins",
            description=(
                "Multi-case lever-origin set. Each key is a case label and the "
                "value is a per-region origin map (region_int -> (x, y, z)). "
                "Each case is run independently end-to-end (its own timeseries "
                "file, its own stats group), so the user can scan over candidate "
                "moment centers (e.g. corners of each container's footprint) and "
                "compare results side by side. Takes precedence over "
                "lever_strategy."
            ),
        ),
    ]

    @model_validator(mode="after")
    def _check_lever_combination(self) -> "MomentBodyConfig":
        if self.lever_origin_cases is not None and self.lever_strategy not in ("fixed",):
            import warnings

            warnings.warn(
                f"MomentBodyConfig {self.name!r}: lever_origin_cases is set, "
                f"so lever_strategy={self.lever_strategy!r} will be ignored. "
                "Set lever_strategy='fixed' (or omit it) to silence this "
                "warning.",
                UserWarning,
                stacklevel=2,
            )
        return self


# ---------------------------------------------------------------------------
# Ce shape coefficient zoning config
# ---------------------------------------------------------------------------


class ExceptionZoningModel(ZoningModel):
    surfaces: Annotated[
        list[str],
        Field(
            ...,
            title="List of surfaces",
            description="Surfaces to include in the exceptional zoning",
        ),
    ]

    @field_validator("surfaces")
    def validate_surface_list(cls, v):
        if len(v) != len(set(v)):
            raise Exception("Invalid exceptions surface list, names must not repeat")
        if len(v) == 0:
            raise Exception("Invalid exceptions surface list, list must not be empty")
        return v


class ZoningConfig(_YamlConfig):
    global_zoning: Annotated[
        ZoningModel,
        Field(
            ...,
            title="Global Zoning Config",
            description="Default Zoning Config applied to all surfaces",
        ),
    ]
    no_zoning: Annotated[
        list[str],
        Field(
            [],
            title="No Zoning Surfaces",
            description="List of surfaces to ignore region mesh generation",
        ),
    ]
    exclude: Annotated[
        list[str],
        Field(
            [],
            title="Surfaces to exclude",
            description="Surfaces to ignore when calculating shape coefficient",
        ),
    ]
    exceptions: Annotated[
        dict[str, ExceptionZoningModel],
        Field(
            {},
            title="Dict with specific zoning",
            description="Specific zoning config that overrides the global zoning",
        ),
    ]

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


# ---------------------------------------------------------------------------
# Cp
# ---------------------------------------------------------------------------


class CpConfig(BasePressureConfig):
    """Configuration to calculate pressure coefficient."""

    timestep_range: Annotated[
        tuple[float, float],
        Field(
            ...,
            title="Timestep Range",
            description="Interval between start and end steps to slice data",
        ),
    ]
    macroscopic_type: Annotated[
        Literal["rho", "pressure"],
        Field(
            "pressure",
            title="Macroscopic type",
            description=(
                "Macroscopic field stored in the body H5. Options: "
                "'pressure' (real pressure, no scaling) or 'rho' (LBM density; "
                "(rho - rho_ref) is converted to pressure via cs^2 = 1/3). "
                "Defaults to 'pressure' when omitted."
            ),
        ),
    ]
    reference_pressure: Annotated[
        Literal["probe", "average"],
        Field(
            "probe",
            title="Reference pressure",
            description=(
                "How p_ref is taken from the reference probe H5 each timestep. "
                "Options: 'probe' (use the first probe point -- the reference "
                "probe placed above the body, the standard wind-tunnel choice) "
                "or 'average' (spatial mean across all probe points at that "
                "timestep). Defaults to 'probe' when omitted."
            ),
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
    simul_U_H: Annotated[
        float,
        Field(
            ...,
            title="Simulation Flow Velocity",
            description="Simulation flow velocity used in the Cp dynamic pressure denominator (always required) and as the time scale L/U when normalize_time=True.",
        ),
    ]
    simul_characteristic_length: Annotated[
        float,
        Field(
            ...,
            title="Simulation Characteristic Length",
            description="Simulation characteristic length used as the time scale L/U when normalize_time=True (otherwise unused).",
        ),
    ]
    normalize_time: Annotated[
        bool,
        Field(
            False,
            title="Normalize time axis",
            description=(
                "When True, the time axis written to /meta/time_normalized in the "
                "Cp output H5 is divided by simul_characteristic_length/simul_U_H "
                "(convective-time normalisation). When False (the default), the "
                "time axis is the raw solver time -- nothing is silently rescaled. "
                "Filters and statistics downstream operate in whichever units this "
                "setting selects, so user-facing windows / durations stay in those "
                "same units."
            ),
        ),
    ]


class CpCaseConfig(_YamlConfig):
    pressure_coefficient: Annotated[
        dict[str, CpConfig],
        Field(
            ...,
            title="Pressure Coefficient configs",
            description="Dictionary with Pressure Coefficient configuration",
        ),
    ]


# ---------------------------------------------------------------------------
# Cf
# ---------------------------------------------------------------------------


class CfConfig(BasePressureConfig):
    """Configuration for force coefficient."""

    bodies: Annotated[
        list[BodyConfig],
        Field(
            ...,
            title="Bodies configuration",
            description="Bodies to process and their zoning config",
        ),
    ]
    nominal_area: Annotated[
        float,
        Field(
            ...,
            gt=0,
            title="Nominal Area",
            description=(
                "Reference area used to non-dimensionalise Cf. Required and must "
                "be > 0: the program does not pick a tribute area for you, since "
                "without an explicit reference area the resulting Cf cannot be "
                "converted back to real-scale forces unambiguously."
            ),
        ),
    ]
    directions: Annotated[
        list[AxisDirections],
        Field(
            ...,
            title="List of directions",
            description="Directions for which force coefficient will be calculated",
        ),
    ]
    transformation: Annotated[
        TransformationConfig,
        Field(
            TransformationConfig(),
            title="Transformation config",
            description="Configuration for mesh transformation",
        ),
    ]


class CfCaseConfig(_YamlConfig):
    bodies: Annotated[
        dict[str, BodyDefinition],
        Field(..., title="Bodies definition", description="Named bodies definition"),
    ]
    force_coefficient: Annotated[
        dict[str, CfConfig],
        Field(
            ...,
            title="Force Coefficient configs",
            description="Dictionary with Force Coefficient configuration",
        ),
    ]

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


# ---------------------------------------------------------------------------
# Cm
# ---------------------------------------------------------------------------


class CmConfig(BasePressureConfig):
    """Configuration for moment coefficient."""

    bodies: Annotated[
        list[MomentBodyConfig],
        Field(
            ...,
            title="Bodies configuration",
            description="Bodies to process and their zoning config",
        ),
    ]
    nominal_volume: Annotated[
        float,
        Field(
            ...,
            gt=0,
            title="Nominal Volume",
            description=(
                "Reference volume used to non-dimensionalise Cm. Required and "
                "must be > 0: the program does not pick a tribute volume for "
                "you, since without an explicit reference volume the resulting "
                "Cm cannot be converted back to real-scale moments unambiguously."
            ),
        ),
    ]
    directions: Annotated[
        list[AxisDirections],
        Field(
            ...,
            title="List of directions",
            description="Directions for which moment coefficient will be calculated",
        ),
    ]
    transformation: Annotated[
        TransformationConfig,
        Field(
            TransformationConfig(),
            title="Transformation config",
            description="Configuration for mesh transformation",
        ),
    ]


class CmCaseConfig(_YamlConfig):
    bodies: Annotated[
        dict[str, BodyDefinition],
        Field(..., title="Bodies definition", description="Named bodies definition"),
    ]
    moment_coefficient: Annotated[
        dict[str, CmConfig],
        Field(
            ...,
            title="Moment Coefficient configs",
            description="Dictionary with Moment Coefficient configuration",
        ),
    ]

    @model_validator(mode="after")
    def validate_body_list(self):
        for body_label in [b.name for cfg in self.moment_coefficient.values() for b in cfg.bodies]:
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


# ---------------------------------------------------------------------------
# Ce (shape coefficient)
# ---------------------------------------------------------------------------


class ZoningBuilder(BaseModel):
    """Reference to an external YAML file containing ZoningConfig."""

    yaml: Annotated[
        str,
        Field(
            ...,
            title="Path to Zoning yaml",
            description="Path to Zoning yaml for constructing zoning configuration",
        ),
    ]
    _base_path: pathlib.Path = pathlib.Path("./")

    def to_zoning_config(self) -> ZoningConfig:
        return ZoningConfig.from_file(self._base_path / pathlib.Path(self.yaml))


class CeConfig(BasePressureConfig):
    """Configuration for shape coefficient."""

    zoning: Annotated[
        ZoningConfig | ZoningBuilder,
        Field(
            ...,
            title="Zoning configuration",
            description="Zoning configuration with intervals information",
        ),
    ]
    sets: Annotated[
        dict[str, list[str]],
        Field(
            {},
            title="Surface sets",
            description="Combine multiple surfaces into a set",
        ),
    ]
    transformation: Annotated[
        TransformationConfig,
        Field(
            TransformationConfig(),
            title="Transformation config",
            description="Configuration for mesh transformation",
        ),
    ]

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


class CeCaseConfig(_YamlConfig):
    shape_coefficient: Annotated[
        dict[str, CeConfig],
        Field(
            ...,
            title="Shape Coefficient configs",
            description="Dictionary of shape coefficient configurations",
        ),
    ]

    @classmethod
    def from_file(cls, filename: pathlib.Path) -> CeCaseConfig:
        cfg = super().from_file(filename)
        for s in cfg.shape_coefficient.values():
            s.zoning._base_path = filename.parent  # type: ignore
            s.to_zoning()
            s.validate_zoning_surfaces()
        return cfg
