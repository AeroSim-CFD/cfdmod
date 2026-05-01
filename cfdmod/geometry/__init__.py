"""Geometry utilities reusable across cfdmod modules.

Currently exposes the triangle-grouping pipeline (``cfdmod.geometry.grouping``).
"""

from cfdmod.geometry.grouping import (
    BySurfaceGrouping,
    ByZoningGrouping,
    ByConnectivityGrouping,
    GroupingSpec,
    GroupingResult,
    apply_groupings,
    dump_groupings,
    load_groupings,
)

__all__ = [
    "BySurfaceGrouping",
    "ByZoningGrouping",
    "ByConnectivityGrouping",
    "GroupingSpec",
    "GroupingResult",
    "apply_groupings",
    "dump_groupings",
    "load_groupings",
]
