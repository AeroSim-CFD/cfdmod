"""Geometry utilities reusable across cfdmod modules.

Exposes the triangle-grouping pipeline (``cfdmod.geometry.grouping``) and
reference-mesh triangle matching (``cfdmod.geometry.matching``).
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
from cfdmod.geometry.matching import (
    MISSING,
    TriangleMatch,
    as_triangle_vertices,
    match_triangles,
)

__all__ = [
    "MISSING",
    "TriangleMatch",
    "as_triangle_vertices",
    "match_triangles",
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
