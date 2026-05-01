"""Group triangles by connected component of the (sub)mesh.

Connectivity is defined by **shared edges**: two triangles are adjacent
when they share a vertex pair. The implementation lives in this module
to keep the ``GroupingSpec`` union complete from step 1.
"""

from __future__ import annotations

from typing import Annotated, Literal

import numpy as np
from lnas import LnasFormat
from pydantic import BaseModel, Field


class ByConnectivityGrouping(BaseModel):
    """Group triangles by connected component (shared-edge adjacency).

    Args:
        kind: Discriminator literal, always ``"by_connectivity"``.
        name_template: Format string for group names. Available
            placeholder: ``{idx}`` (component index, 0-based; components
            are ordered by descending triangle count so ``cc0`` is the
            largest).
        min_triangles: Components smaller than this are dropped.
        restrict_to: Optional list of earlier group names; when set, only
            triangles in (the union of) those groups participate in the
            connectivity analysis. Edges to triangles outside the
            restriction are ignored.
    """

    kind: Literal["by_connectivity"] = "by_connectivity"
    name_template: Annotated[
        str,
        Field("cc{idx}", description="Format string for group names; placeholder: {idx}"),
    ]
    min_triangles: Annotated[
        int,
        Field(1, ge=1, description="Drop components with fewer triangles than this"),
    ]
    restrict_to: Annotated[
        list[str] | None,
        Field(None, description="Optional list of earlier group names to restrict to."),
    ]


def apply_by_connectivity(
    spec: ByConnectivityGrouping,
    mesh: LnasFormat,
    allowed: np.ndarray | None,
) -> dict[str, np.ndarray]:
    """Connected-components grouping. See module docstring."""
    raise NotImplementedError(
        "ByConnectivityGrouping is declared in step 1 but implemented in step 2 "
        "of the triangle-grouping pipeline refactor (issue #128)."
    )
