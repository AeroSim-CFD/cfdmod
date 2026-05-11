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
    # Pressure (v2 entry points removed in v3; use `cfdmod run <template.yaml>`
    # or the v3 recipes -- see notebooks/tutorials/ and
    # fixtures/tests/pressure/templates/).
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
    # v3 paradigm: data sources, ops, recipes (issue #131)
    "DataSource",
    "SurfaceDataSource",
    "VolumeDataSource",
    "PointsDataSource",
    "GroupsDataSource",
    "ModesDataSource",
    "TimeAxis",
    "Topology",
    "ElementMeta",
    "Grouping",
    "FieldMeta",
    "Container",
    "Pipeline",
    "compose",
    "MemoryStorage",
    "MemoryFieldStore",
    "XdmfH5Storage",
    "H5FieldStore",
    "core_ops",
    "recipes",
    "load_template",
    "run_template",
    "PipelineTemplate",
    "register_op",
    "OP_REGISTRY",
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

# v3 paradigm exports (issue #131). The legacy public symbols above are
# unchanged; the new names live next to them and become the canonical
# entry points for new code.
from cfdmod.adapters.memory import MemoryFieldStore, MemoryStorage
from cfdmod.adapters.xdmf_h5 import H5FieldStore, XdmfH5Storage
from cfdmod.core import (
    OP_REGISTRY,
    Container,
    DataSource,
    ElementMeta,
    FieldMeta,
    Grouping,
    GroupsDataSource,
    ModesDataSource,
    Pipeline,
    PipelineTemplate,
    PointsDataSource,
    SurfaceDataSource,
    TimeAxis,
    Topology,
    VolumeDataSource,
    compose,
    load_template,
    register_op,
    run_template,
)
from cfdmod.core import ops as core_ops
from cfdmod.core import recipes

# Expose cfdmod.recipes and cfdmod.ops as importable submodule paths so
# users can write `from cfdmod.recipes import build_cp` without reaching
# into cfdmod.core.
import sys as _sys

_sys.modules.setdefault("cfdmod.recipes", recipes)
_sys.modules.setdefault("cfdmod.ops", core_ops)
