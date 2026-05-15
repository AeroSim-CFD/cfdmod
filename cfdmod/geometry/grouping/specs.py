"""Discriminated union of grouping specs.

Add a new kind by:

1. Creating ``cfdmod/geometry/grouping/kinds/<kind>.py`` that defines
   ``<Kind>Grouping(BaseModel)`` with ``kind: Literal['<kind>']`` and an
   ``apply_<kind>(spec, mesh, allowed) -> dict[str, np.ndarray]``.
2. Adding the spec class to the ``GroupingSpec`` union below.
3. Adding a dispatch branch in ``cfdmod.geometry.grouping.base._dispatch``.
"""

from __future__ import annotations

from typing import Annotated

from pydantic import Field

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

GroupingSpec = Annotated[
    BySurfaceGrouping
    | ByZoningGrouping
    | ByDivisionsGrouping
    | BySizeGrouping
    | ByConnectivityGrouping
    | ByNormalGrouping
    | ByPlaneGrouping
    | ByPercentileGrouping
    | ByCylindricalGrouping
    | CustomGrouping,
    Field(discriminator="kind"),
]
