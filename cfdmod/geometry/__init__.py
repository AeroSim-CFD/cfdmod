"""Geometry utilities reusable across cfdmod modules.

Exposes the triangle-grouping pipeline (``cfdmod.geometry.grouping``),
reference-mesh triangle matching (``cfdmod.geometry.matching``) and the
projected (shadow) area of a surface along a wind direction
(``cfdmod.geometry.projection``).
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
from cfdmod.geometry.projection import ProjectedArea, projected_area

__all__ = [
    "MISSING",
    "TriangleMatch",
    "as_triangle_vertices",
    "match_triangles",
    "ProjectedArea",
    "projected_area",
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
