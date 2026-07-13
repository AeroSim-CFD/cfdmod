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
    requires_element_meta: ClassVar[frozenset[str]] = frozenset({"area", "normal"})

    def produced_fields(self) -> frozenset[str]:
        return frozenset(f"{self.out_prefix}_{d}" for d in self.directions)


def force_contribution(ds: DataSource, p: ForceContributionParams) -> DataSource:
    if ds.elements.area is None or ds.elements.normal is None:
        raise ValueError(
            "force_contribution requires elements.area and elements.normal; "
            "attach them with mesh_attach first."
        )

    # Preserve the Cp field's dtype (float32 for solver output, float64 for
    # float64 sources) rather than upcasting; cast the geometric factor to match
    # so a float64 area does not silently promote a float32 result back up.
    cp = np.asarray(ds.fields.read(p.field))
    if cp.ndim != 2:
        raise ValueError(
            f"field {p.field!r} must be 2-D (n_elements, n_timesteps); got {cp.shape}"
        )

    dt = cp.dtype
    area = np.asarray(ds.elements.area, dtype=dt)
    normals = np.asarray(ds.elements.normal, dtype=dt)
    axis_map = {"x": 0, "y": 1, "z": 2}

    out = ds
    for d in p.directions:
        axis = axis_map[d]
        # cf_dir[tri, t] = -cp[tri, t] * area[tri] * normal_dir[tri] / nominal_area
        cf = -cp * (area * normals[:, axis])[:, None] / p.nominal_area
        name = f"{p.out_prefix}_{d}"
        out = out.with_field(name, cf, meta=FieldMeta(name=name, unit="-"))
    return out
