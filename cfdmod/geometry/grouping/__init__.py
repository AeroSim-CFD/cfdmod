"""Triangle-grouping pipeline.

A ``GroupingSpec`` describes one operation that partitions or selects
triangles of a parent ``LnasFormat`` mesh into named groups. Specs are
composed left-to-right via :func:`apply_groupings`, which produces a
:class:`GroupingResult` mapping ``group_name -> triangle_indices``.

Properties of the abstraction:

- A triangle may appear in **zero, one, or many** groups.
- Specs are Pydantic models in a discriminated union, dispatched by ``kind``.
- Adding a new grouping kind is a localised change: define a new spec +
  ``apply`` function under :mod:`cfdmod.geometry.grouping.kinds` and add
  it to the ``GroupingSpec`` union in :mod:`cfdmod.geometry.grouping.specs`.

This module mirrors the architecture of
:mod:`cfdmod.pressure.filters` for time-series pipelines.
"""

from cfdmod.geometry.grouping.base import GroupingResult, apply_groupings
from cfdmod.geometry.grouping.io import dump_groupings, load_groupings
from cfdmod.geometry.grouping.kinds.by_connectivity import ByConnectivityGrouping
from cfdmod.geometry.grouping.kinds.by_custom import CustomGrouping
from cfdmod.geometry.grouping.kinds.by_cylindrical import ByCylindricalGrouping
from cfdmod.geometry.grouping.kinds.by_divisions import ByDivisionsGrouping
from cfdmod.geometry.grouping.kinds.by_normal import ByNormalGrouping
from cfdmod.geometry.grouping.kinds.by_percentile import ByPercentileGrouping
from cfdmod.geometry.grouping.kinds.by_plane import ByPlaneGrouping
from cfdmod.geometry.grouping.kinds.by_size import BySizeGrouping
from cfdmod.geometry.grouping.kinds.by_surface import BySurfaceGrouping
from cfdmod.geometry.grouping.kinds.by_zoning import ByZoningGrouping
from cfdmod.geometry.grouping.regroup import (
    BySizeRoundedPerComponent,
    RegroupSpec,
    expand_size_rounded_chain,
)
from cfdmod.geometry.grouping.specs import GroupingSpec

__all__ = [
    "GroupingResult",
    "GroupingSpec",
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
    "BySizeRoundedPerComponent",
    "RegroupSpec",
    "apply_groupings",
    "dump_groupings",
    "load_groupings",
    "expand_size_rounded_chain",
]
