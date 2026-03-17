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
]

from cfdmod.use_cases.loft import LoftParams, LoftCaseConfig, generate_loft_surface
from cfdmod.use_cases.roughness_gen import (
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
from cfdmod.use_cases.pressure.base_config import BasePressureConfig
from cfdmod.use_cases.pressure.cp_config import CpConfig, CpCaseConfig
from cfdmod.use_cases.pressure.cp_data import process_cp
from cfdmod.use_cases.pressure.force import CfConfig, CfCaseConfig, process_Cf
from cfdmod.use_cases.pressure.moment import CmConfig, CmCaseConfig, process_Cm
from cfdmod.use_cases.pressure.shape import CeConfig, CeCaseConfig
from cfdmod.use_cases.pressure.shape.Ce_data import process_Ce
from cfdmod.use_cases.pressure.zoning import (
    ZoningModel,
    BodyDefinition,
    BodyConfig,
    MomentBodyConfig,
)
from cfdmod.use_cases.s1 import (
    Profile,
    EUCat,
    NBRCat,
    get_EU_u_profile,
    get_NBR_u_profile,
    get_EU_cat_u_profile,
    get_NBR_cat_u_profile,
    S1Probe,
)
from cfdmod.use_cases.climate import WindProfile, fit_weibull, directional_weibull_fit
from cfdmod.use_cases.climate import fit_gumbel, directional_gumbel_fit
from cfdmod.use_cases.analytical import WindProfile_NBR, WindProfile_EU
from cfdmod.analysis.inflow import NormalizationParameters, InflowData
from cfdmod.api.configs import HashableConfig
from cfdmod.api.geometry import read_stl, export_stl
