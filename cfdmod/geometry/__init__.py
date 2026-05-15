"""Geometry utilities reusable across cfdmod modules.

Currently exposes the triangle-grouping pipeline (``cfdmod.geometry.grouping``).
"""

from cfdmod.geometry.grouping import (
    BySurfaceGrouping,
    ByZoningGrouping,
    ByDivisionsGrouping,
    BySizeGrouping,
    BySizeRoundedPerComponent,
    ByConnectivityGrouping,
    ByNormalGrouping,
    ByPlaneGrouping,
    ByPercentileGrouping,
    ByCylindricalGrouping,
    CustomGrouping,
    GroupingSpec,
    GroupingResult,
    RegroupSpec,
    apply_groupings,
    dump_groupings,
    expand_size_rounded_chain,
    load_groupings,
)

__all__ = [
    "BySurfaceGrouping",
    "ByZoningGrouping",
    "ByDivisionsGrouping",
    "BySizeGrouping",
    "BySizeRoundedPerComponent",
    "ByConnectivityGrouping",
    "ByNormalGrouping",
    "ByPlaneGrouping",
    "ByPercentileGrouping",
    "ByCylindricalGrouping",
    "CustomGrouping",
    "GroupingSpec",
    "GroupingResult",
    "RegroupSpec",
    "apply_groupings",
    "dump_groupings",
    "expand_size_rounded_chain",
    "load_groupings",
]
