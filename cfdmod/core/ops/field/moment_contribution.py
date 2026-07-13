"""Per-element moment contribution: cf -> cm_x / cm_y / cm_z.

The math, mirroring :func:`cfdmod.pressure.functions.transform_Cm`:

    r[tri]              = centroid[tri] - lever_origin
    cf[tri, t, :]       = (cf_x[tri, t], cf_y[tri, t], cf_z[tri, t])
    moment[tri, t, :]   = r[tri] x cf[tri, t, :]
    cm_dir[tri, t]      = moment[tri, t, dir] * nominal_area / nominal_volume

The nominal_area factor undoes the Cf normalisation so the moment
output is non-dimensional with respect to nominal_volume only -- this
matches the cfdmod convention.

Force contributions (``cf_x`` / ``cf_y`` / ``cf_z``) must already be
attached by :func:`force_contribution`. Centroids come from
``elements.position`` (populated by :func:`mesh_attach`).
"""

from __future__ import annotations

__all__ = ["MomentContributionParams", "moment_contribution"]

from typing import ClassVar, Literal

import numpy as np
from pydantic import Field

from cfdmod.core.data_source import DataSource
from cfdmod.core.field_meta import FieldMeta
from cfdmod.core.ops import OpParams


class MomentContributionParams(OpParams):
    """Parameters for :func:`moment_contribution`.

    Attributes:
        lever_origin: ``(x, y, z)`` coordinate to take moments about.
        nominal_area: The Cf normalisation factor; needed to back out
            the raw per-element forces.
        nominal_volume: Volume used to non-dimensionalise the moment.
        directions: Subset of ``("x", "y", "z")`` to compute.
        in_prefix: Force field prefix on the source. Defaults to
            ``"cf"`` so the op reads ``cf_x``, ``cf_y``, ``cf_z``.
        out_prefix: Output field prefix. Defaults to ``"cm"``.
    """

    kind: Literal["moment_contribution"] = "moment_contribution"
    lever_origin: tuple[float, float, float]
    nominal_area: float = Field(gt=0)
    nominal_volume: float = Field(gt=0)
    directions: list[Literal["x", "y", "z"]] = ["x", "y", "z"]
    in_prefix: str = "cf"
    out_prefix: str = "cm"

    chunkable_along: ClassVar[frozenset[str]] = frozenset({"time"})
    requires_element_meta: ClassVar[frozenset[str]] = frozenset({"position"})

    def consumed_fields(self) -> frozenset[str]:
        return frozenset(f"{self.in_prefix}_{d}" for d in self.directions)

    def produced_fields(self) -> frozenset[str]:
        return frozenset(f"{self.out_prefix}_{d}" for d in self.directions)


def moment_contribution(ds: DataSource, p: MomentContributionParams) -> DataSource:
    if ds.elements.position is None:
        raise ValueError(
            "moment_contribution requires elements.position; "
            "attach centroids with mesh_attach first."
        )

    # Follow the Cf fields' dtype (see force_contribution) so float32 forces stay
    # float32 through the moment; cast the lever arm to match.
    cf_arrays = {d: np.asarray(ds.fields.read(f"{p.in_prefix}_{d}")) for d in ("x", "y", "z")}
    dt = cf_arrays["x"].dtype
    centroids = np.asarray(ds.elements.position, dtype=dt)
    r = centroids - np.asarray(p.lever_origin, dtype=dt)[None, :]

    # Undo Cf's nominal-area normalisation -> per-element force.
    fx = cf_arrays["x"] * p.nominal_area
    fy = cf_arrays["y"] * p.nominal_area
    fz = cf_arrays["z"] * p.nominal_area

    # m = r x f, component-wise.
    rx, ry, rz = r[:, 0:1], r[:, 1:2], r[:, 2:3]
    out = ds
    if "x" in p.directions:
        mx = (ry * fz - rz * fy) / p.nominal_volume
        out = out.with_field(f"{p.out_prefix}_x", mx, meta=FieldMeta(name=f"{p.out_prefix}_x"))
    if "y" in p.directions:
        my = (rz * fx - rx * fz) / p.nominal_volume
        out = out.with_field(f"{p.out_prefix}_y", my, meta=FieldMeta(name=f"{p.out_prefix}_y"))
    if "z" in p.directions:
        mz = (rx * fy - ry * fx) / p.nominal_volume
        out = out.with_field(f"{p.out_prefix}_z", mz, meta=FieldMeta(name=f"{p.out_prefix}_z"))
    return out
