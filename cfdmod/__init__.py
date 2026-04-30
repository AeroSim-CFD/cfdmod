"""aerosim-cfdmod: Post-processing and geometry preparation for CFD wind tunnel simulations."""

__all__ = [
    # Loft
    "LoftParams",
    "LoftCaseConfig",
    "generate_loft_surface",
    # Roughness
    "ElementParams",
    "SpacingParams",
    "BoundingBox",
    "PositionParams",
    "RadialParams",
    "GenerationParams",
    "build_single_element",
    "linear_pattern",
    "radial_pattern",
    # Pressure base
    "BasePressureConfig",
    # Pressure Cp
    "CpConfig",
    "CpCaseConfig",
    "run_cp",
    # Pressure Cf
    "CfConfig",
    "CfCaseConfig",
    "process_Cf",
    "run_cf",
    # Pressure Cm
    "CmConfig",
    "CmCaseConfig",
    "process_Cm",
    "run_cm",
    # Pressure Ce
    "CeConfig",
    "CeCaseConfig",
    "process_Ce",
    "run_ce",
    # Filters
    "MovingAverageFilter",
    "FilterSpec",
    "apply_filters",
    # Zoning
    "ZoningModel",
    "BodyDefinition",
    "BodyConfig",
    "MomentBodyConfig",
    # S1
    "Profile",
    "EUCat",
    "NBRCat",
    "get_EU_u_profile",
    "get_NBR_u_profile",
    "get_EU_cat_u_profile",
    "get_NBR_cat_u_profile",
    "S1Probe",
    # Climate
    "WindProfile",
    "fit_weibull",
    "directional_weibull_fit",
    "fit_gumbel",
    "directional_gumbel_fit",
    # Analytical
    "WindProfile_NBR",
    "WindProfile_EU",
    # Inflow
    "NormalizationParameters",
    "InflowData",
    # IO
    "read_stl",
    "export_stl",
    "load_mesh",
    "mesh_from_h5",
    "read_processing_metadata",
    "write_processing_metadata",
    "read_timeseries_df",
    "plot_timeseries",
    "to_csv",
    # Notebook utils
    "mesh_summary",
    "show_config",
    "load_lnas",
]

from cfdmod.inflow import InflowData, NormalizationParameters
from cfdmod.analytical import WindProfile_EU, WindProfile_NBR
from cfdmod.climate import (
    WindProfile,
    directional_gumbel_fit,
    directional_weibull_fit,
    fit_gumbel,
    fit_weibull,
)
from cfdmod.io import (
    export_stl,
    load_mesh,
    mesh_from_h5,
    plot_timeseries,
    read_processing_metadata,
    read_stl,
    read_timeseries_df,
    to_csv,
    write_processing_metadata,
)
from cfdmod.loft import LoftCaseConfig, LoftParams, generate_loft_surface
from cfdmod.notebook_utils import load_lnas, mesh_summary, show_config
from cfdmod.pressure.filters import FilterSpec, MovingAverageFilter, apply_filters
from cfdmod.pressure.functions import process_Ce, process_Cf, process_Cm
from cfdmod.pressure.parameters import (
    BasePressureConfig,
    BodyConfig,
    BodyDefinition,
    CeCaseConfig,
    CeConfig,
    CfCaseConfig,
    CfConfig,
    CmCaseConfig,
    CmConfig,
    CpCaseConfig,
    CpConfig,
    MomentBodyConfig,
    ZoningModel,
)
from cfdmod.pressure.run import run_ce, run_cf, run_cm, run_cp
from cfdmod.roughness import (
    BoundingBox,
    ElementParams,
    GenerationParams,
    PositionParams,
    RadialParams,
    SpacingParams,
    build_single_element,
    linear_pattern,
    radial_pattern,
)
from cfdmod.s1 import (
    EUCat,
    NBRCat,
    Profile,
    S1Probe,
    get_EU_cat_u_profile,
    get_EU_u_profile,
    get_NBR_cat_u_profile,
    get_NBR_u_profile,
)
