"""aerosim-cfdmod: Post-processing and geometry preparation for CFD wind tunnel simulations."""

from __future__ import annotations

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
    # Geometry grouping (canonical triangle-grouping pipeline -- the v3
    # ops in cfdmod.core.ops.geometric delegate to apply_groupings).
    "BySurfaceGrouping",
    "ByZoningGrouping",
    "ByDivisionsGrouping",
    "BySizeGrouping",
    "ByConnectivityGrouping",
    "ByNormalGrouping",
    "ByPlaneGrouping",
    "ByPercentileGrouping",
    "ByCylindricalGrouping",
    "CustomGrouping",
    "GroupingSpec",
    "GroupingResult",
    "BySizeRoundedPerComponent",
    "RegroupSpec",
    "apply_groupings",
    "dump_groupings",
    "load_groupings",
    "expand_size_rounded_chain",
    # Topology regrouping op (v3)
    "RegroupTopologyParams",
    "regroup_topology",
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
    "MemoryBlobStore",
    "XdmfH5Storage",
    "H5FieldStore",
    "XdmfH5BlobStorage",
    "BlobStore",
    "core_ops",
    "recipes",
    "load_template",
    "run_template",
    "PipelineTemplate",
    "register_op",
    "OP_REGISTRY",
    "OpInfo",
    "list_ops",
    "op_info",
    # Freshness / output staleness
    "output_status",
    "OutputStatus",
    "FreshnessConfig",
    # Errors (issue #147)
    "CfdmodError",
    "TemplateError",
    "TemplateReferenceError",
    "OpError",
    "StorageKeyError",
    # Regroup
    "RegroupConfig",
    "RegroupIndex",
    "build_regroup_mapping",
    "build_regrouped_mesh",
    "apply_regroup_to_timeseries",
    "expand_regroup_chain",
    "run_regroup",
    # Remesh
    "merge_coplanar",
    "decimate_qem",
    "remesh_per_group",
]

# ---------------------------------------------------------------------------
# Lazy import surface (issue #147)
# ---------------------------------------------------------------------------
#
# Public symbols are bound on first access (PEP 562 module __getattr__)
# rather than eagerly at package import. This keeps `import cfdmod` -- and
# therefore importing the v3 schema / op catalog under cfdmod.core -- free of
# the heavy scientific stack (h5py, matplotlib, pandas, pyarrow, vtk,
# trimesh). A service consumer that only needs the template schema and op
# catalog pays for none of it; those deps load only when a symbol that needs
# them is first touched.

import importlib
import sys as _sys
from typing import Any

# symbol name -> module that defines it.
_SYMBOL_MODULE: dict[str, str] = {
    # Loft
    "LoftParams": "cfdmod.loft",
    "LoftCaseConfig": "cfdmod.loft",
    "generate_loft_surface": "cfdmod.loft",
    # Roughness
    "ElementParams": "cfdmod.roughness",
    "SpacingParams": "cfdmod.roughness",
    "BoundingBox": "cfdmod.roughness",
    "PositionParams": "cfdmod.roughness",
    "RadialParams": "cfdmod.roughness",
    "GenerationParams": "cfdmod.roughness",
    "build_single_element": "cfdmod.roughness",
    "linear_pattern": "cfdmod.roughness",
    "radial_pattern": "cfdmod.roughness",
    # Geometry grouping
    "BySurfaceGrouping": "cfdmod.geometry",
    "ByZoningGrouping": "cfdmod.geometry",
    "ByDivisionsGrouping": "cfdmod.geometry",
    "BySizeGrouping": "cfdmod.geometry",
    "ByConnectivityGrouping": "cfdmod.geometry",
    "ByNormalGrouping": "cfdmod.geometry",
    "ByPlaneGrouping": "cfdmod.geometry",
    "ByPercentileGrouping": "cfdmod.geometry",
    "ByCylindricalGrouping": "cfdmod.geometry",
    "CustomGrouping": "cfdmod.geometry",
    "GroupingSpec": "cfdmod.geometry",
    "GroupingResult": "cfdmod.geometry",
    "BySizeRoundedPerComponent": "cfdmod.geometry",
    "RegroupSpec": "cfdmod.geometry",
    "apply_groupings": "cfdmod.geometry",
    "dump_groupings": "cfdmod.geometry",
    "load_groupings": "cfdmod.geometry",
    "expand_size_rounded_chain": "cfdmod.geometry",
    # Topology regrouping op (v3)
    "RegroupTopologyParams": "cfdmod.core.ops.geometric",
    "regroup_topology": "cfdmod.core.ops.geometric",
    # S1
    "Profile": "cfdmod.s1",
    "EUCat": "cfdmod.s1",
    "NBRCat": "cfdmod.s1",
    "get_EU_u_profile": "cfdmod.s1",
    "get_NBR_u_profile": "cfdmod.s1",
    "get_EU_cat_u_profile": "cfdmod.s1",
    "get_NBR_cat_u_profile": "cfdmod.s1",
    "S1Probe": "cfdmod.s1",
    # Climate
    "WindProfile": "cfdmod.climate",
    "fit_weibull": "cfdmod.climate",
    "directional_weibull_fit": "cfdmod.climate",
    "fit_gumbel": "cfdmod.climate",
    "directional_gumbel_fit": "cfdmod.climate",
    # Analytical
    "WindProfile_NBR": "cfdmod.analytical",
    "WindProfile_EU": "cfdmod.analytical",
    # Inflow
    "NormalizationParameters": "cfdmod.inflow",
    "InflowData": "cfdmod.inflow",
    # IO
    "read_stl": "cfdmod.io",
    "export_stl": "cfdmod.io",
    "load_mesh": "cfdmod.io",
    "mesh_from_h5": "cfdmod.io",
    "read_processing_metadata": "cfdmod.io",
    "write_processing_metadata": "cfdmod.io",
    "read_timeseries_df": "cfdmod.io",
    "plot_timeseries": "cfdmod.io",
    "to_csv": "cfdmod.io",
    # Notebook utils
    "mesh_summary": "cfdmod.notebook_utils",
    "show_config": "cfdmod.notebook_utils",
    "load_lnas": "cfdmod.notebook_utils",
    # v3 value objects / composition / catalog (cfdmod.core imports light)
    "DataSource": "cfdmod.core",
    "SurfaceDataSource": "cfdmod.core",
    "VolumeDataSource": "cfdmod.core",
    "PointsDataSource": "cfdmod.core",
    "GroupsDataSource": "cfdmod.core",
    "ModesDataSource": "cfdmod.core",
    "TimeAxis": "cfdmod.core",
    "Topology": "cfdmod.core",
    "ElementMeta": "cfdmod.core",
    "Grouping": "cfdmod.core",
    "FieldMeta": "cfdmod.core",
    "Container": "cfdmod.core",
    "Pipeline": "cfdmod.core",
    "compose": "cfdmod.core",
    "load_template": "cfdmod.core",
    "run_template": "cfdmod.core",
    "PipelineTemplate": "cfdmod.core",
    "register_op": "cfdmod.core",
    "OP_REGISTRY": "cfdmod.core",
    "OpInfo": "cfdmod.core",
    "list_ops": "cfdmod.core",
    "op_info": "cfdmod.core",
    "output_status": "cfdmod.core",
    "OutputStatus": "cfdmod.core",
    "FreshnessConfig": "cfdmod.core",
    # Errors
    "CfdmodError": "cfdmod.core",
    "TemplateError": "cfdmod.core",
    "TemplateReferenceError": "cfdmod.core",
    "OpError": "cfdmod.core",
    "StorageKeyError": "cfdmod.core",
    "BlobStore": "cfdmod.core",
    # Storage adapters (pull h5py -- lazy on purpose)
    "MemoryStorage": "cfdmod.adapters.memory",
    "MemoryFieldStore": "cfdmod.adapters.memory",
    "MemoryBlobStore": "cfdmod.adapters.memory",
    "XdmfH5Storage": "cfdmod.adapters.xdmf_h5",
    "H5FieldStore": "cfdmod.adapters.xdmf_h5",
    "XdmfH5BlobStorage": "cfdmod.adapters.xdmf_h5",
    # Regroup
    "RegroupConfig": "cfdmod.regroup",
    "RegroupIndex": "cfdmod.regroup",
    "build_regroup_mapping": "cfdmod.regroup",
    "build_regrouped_mesh": "cfdmod.regroup",
    "apply_regroup_to_timeseries": "cfdmod.regroup",
    "expand_regroup_chain": "cfdmod.regroup",
    "run_regroup": "cfdmod.regroup",
    # Remesh
    "merge_coplanar": "cfdmod.remesh",
    "decimate_qem": "cfdmod.remesh",
    "remesh_per_group": "cfdmod.remesh",
}

# Public names that resolve to a whole submodule rather than a symbol.
_SUBMODULE_ATTRS: dict[str, str] = {
    "core_ops": "cfdmod.core.ops",
    "recipes": "cfdmod.core.recipes",
}


def __getattr__(name: str) -> Any:
    module_path = _SYMBOL_MODULE.get(name)
    if module_path is not None:
        value = getattr(importlib.import_module(module_path), name)
        globals()[name] = value  # cache so subsequent access skips __getattr__
        return value
    submodule_path = _SUBMODULE_ATTRS.get(name)
    if submodule_path is not None:
        module = importlib.import_module(submodule_path)
        globals()[name] = module
        return module
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__() -> list[str]:
    return sorted(set(__all__) | set(globals()))


# Expose cfdmod.recipes and cfdmod.ops as importable submodule paths so users
# can write `from cfdmod.recipes import build_cp` without reaching into
# cfdmod.core. These modules import light (no h5py at module top), so
# registering the aliases here does not defeat the lazy surface above.
_sys.modules.setdefault("cfdmod.recipes", importlib.import_module("cfdmod.core.recipes"))
_sys.modules.setdefault("cfdmod.ops", importlib.import_module("cfdmod.core.ops"))
