"""Per-element force contribution: cp -> cf_x / cf_y / cf_z.

The math, mirroring :func:`cfdmod.pressure.functions.transform_Cf`:

    cf_dir[t, tri] = -cp[t, tri] * area[tri] * normal_dir[tri] / nominal_area

Areas and normals come from the source's :class:`ElementMeta` (set by
the :func:`mesh_attach` op). The three new fields land on the same
source; downstream aggregation (typically
:func:`field_series_for_groups` with ``agg="sum"``) sums them per
body / region to yield the per-body force coefficients.

The minus sign matches the cfdmod convention: positive ``cf_x``
corresponds to a body-frame force in the +x direction induced by a
positive Cp.
"""

from __future__ import annotations

__all__ = ["ForceContributionParams", "force_contribution"]

from typing import ClassVar, Literal

import numpy as np
from pydantic import Field

from cfdmod.core.data_source import DataSource
from cfdmod.core.field_meta import FieldMeta
from cfdmod.core.ops import OpParams


class ForceContributionParams(OpParams):
    """Parameters for :func:`force_contribution`.

    Attributes:
        field: Source field name carrying Cp. Defaults to ``"cp"``.
        nominal_area: Reference area used to non-dimensionalise the
            force.
        directions: Subset of ``("x", "y", "z")`` to compute.
        out_prefix: Output field prefix. Defaults to ``"cf"`` so the
            resulting fields are ``cf_x`` / ``cf_y`` / ``cf_z``.
    """

    kind: Literal["force_contribution"] = "force_contribution"
    field: str = "cp"
    nominal_area: float = Field(gt=0)
    directions: list[Literal["x", "y", "z"]] = ["x", "y", "z"]
    out_prefix: str = "cf"

    chunkable_along: ClassVar[frozenset[str]] = frozenset({"time"})


def force_contribution(ds: DataSource, p: ForceContributionParams) -> DataSource:
    if ds.elements.area is None or ds.elements.normal is None:
        raise ValueError(
            "force_contribution requires elements.area and elements.normal; "
            "attach them with mesh_attach first."
        )

    cp = np.asarray(ds.fields.read(p.field), dtype=np.float64)
    if cp.ndim != 2:
        raise ValueError(
            f"field {p.field!r} must be 2-D (n_elements, n_timesteps); got {cp.shape}"
        )

    area = ds.elements.area
    normals = ds.elements.normal
    axis_map = {"x": 0, "y": 1, "z": 2}

    out = ds
    for d in p.directions:
        axis = axis_map[d]
        # cf_dir[tri, t] = -cp[tri, t] * area[tri] * normal_dir[tri] / nominal_area
        cf = -cp * (area * normals[:, axis])[:, None] / p.nominal_area
        name = f"{p.out_prefix}_{d}"
        out = out.with_field(name, cf, meta=FieldMeta(name=name, unit="-"))
    return out
