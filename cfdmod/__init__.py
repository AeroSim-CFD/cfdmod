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
    "process_cp",
    # Pressure Cf
    "CfConfig",
    "CfCaseConfig",
    "process_Cf",
    # Pressure Cm
    "CmConfig",
    "CmCaseConfig",
    "process_Cm",
    # Pressure Ce
    "CeConfig",
    "CeCaseConfig",
    "process_Ce",
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
    # API
    "HashableConfig",
    "read_stl",
    "export_stl",
    # Notebook utils
    "mesh_summary",
    "show_config",
    "load_lnas",
]

from cfdmod.loft import LoftParams, LoftCaseConfig, generate_loft_surface
from cfdmod.roughness import (
    ElementParams,
    SpacingParams,
    BoundingBox,
    PositionParams,
    RadialParams,
    GenerationParams,
    build_single_element,
    linear_pattern,
    radial_pattern,
)
from cfdmod.pressure.parameters import (
    BasePressureConfig,
    CpConfig,
    CpCaseConfig,
    CfConfig,
    CfCaseConfig,
    CmConfig,
    CmCaseConfig,
    CeConfig,
    CeCaseConfig,
    ZoningModel,
    BodyDefinition,
    BodyConfig,
    MomentBodyConfig,
)
from cfdmod.pressure.functions import process_Cf, process_Cm, process_Ce
from cfdmod.pressure.run import run_cp as process_cp
from cfdmod.s1 import (
    Profile,
    EUCat,
    NBRCat,
    get_EU_u_profile,
    get_NBR_u_profile,
    get_EU_cat_u_profile,
    get_NBR_cat_u_profile,
    S1Probe,
)
from cfdmod.climate import WindProfile, fit_weibull, directional_weibull_fit
from cfdmod.climate import fit_gumbel, directional_gumbel_fit
from cfdmod.analytical import WindProfile_NBR, WindProfile_EU
from cfdmod.analysis.inflow import NormalizationParameters, InflowData
from cfdmod.config import HashableConfig
from cfdmod.io import read_stl, export_stl
from cfdmod.notebook_utils import mesh_summary, show_config, load_lnas
