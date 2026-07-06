"""S1 recipe -- velocity-profile ratio against a reference.

Per the odt::

    volume field + probe positions  -> extracted velocity profiles
    profiles + interpolation scheme -> profiles on common heights
    profiles / reference profiles   -> S1

The recipe takes a :class:`PointsDataSource` (the CFD profile sampled
along a vertical column) and a reference :class:`PointsDataSource`
(the standard / target profile). It interpolates the CFD profile onto
the reference heights, then divides element-wise (broadcasting rule 4
for time-resolved profiles, rule 3 row-wise when one side is
time-aggregated). Wall samples are dropped.
"""

from __future__ import annotations

__all__ = ["S1RecipeConfig", "s1_pipeline", "build_s1"]

import numpy as np
from pydantic import BaseModel, ConfigDict

from cfdmod.adapters.memory import MemoryFieldStore
from cfdmod.core import algebra
from cfdmod.core.data_source import PointsDataSource
from cfdmod.core.ops.data_source_create.profile_interpolation import (
    ProfileInterpolationParams,
    profile_interpolation,
)
from cfdmod.core.topology import ElementMeta, Topology


class S1RecipeConfig(BaseModel):
    """S1 recipe parameters.

    Attributes:
        field: Velocity field on both profiles. Defaults to ``"u"``.
        out: Output field name. Defaults to ``"s1"``.
        wall_threshold: Reference values whose absolute value falls
            below this threshold (and the wall row at z=0) are dropped
            from the output. Mirrors the legacy ``Profile.__truediv__``
            check.
    """

    model_config = ConfigDict(frozen=True)

    field: str = "u"
    out: str = "s1"
    wall_threshold: float = 1e-6


def build_s1(
    cfd_profile: PointsDataSource,
    reference: PointsDataSource,
    cfg: S1RecipeConfig,
) -> PointsDataSource:
    """Compute the S1 profile.

    The reference's heights determine the common axis; the CFD profile
    is reinterpolated onto them.
    """
    ref_z = reference.elements.position[:, 2]

    cfd_on_ref = profile_interpolation(
        cfd_profile,
        ProfileInterpolationParams(target_heights=ref_z, field=cfg.field),
    )

    # Reference may carry a wall sample at z=0 that we drop afterwards; the
    # division warning would be noise.
    with np.errstate(divide="ignore", invalid="ignore"):
        s1_full = algebra.div(cfd_on_ref, reference, field=cfg.field, out=cfg.out)

    ref_arr = reference.fields.read(cfg.field)
    if ref_arr.ndim == 2:
        # mask = column-wise OR of "above threshold" -- conservatively
        # keep heights where the reference is non-trivial at *any* time.
        mask = (np.abs(ref_arr) > cfg.wall_threshold).any(axis=1)
    else:
        mask = np.abs(ref_arr) > cfg.wall_threshold
    if ref_z.size > 0 and ref_z[0] == 0.0:
        mask[0] = False

    if not mask.any():
        raise ValueError("S1: every reference sample fell below the wall threshold")

    keep = np.flatnonzero(mask)
    s1_arr = s1_full.fields.read(cfg.out)[keep]
    new_pos = s1_full.elements.position[keep]

    return PointsDataSource(
        time=s1_full.time,
        topology=Topology.points(new_pos),
        elements=ElementMeta(position=new_pos),
        fields=MemoryFieldStore({cfg.out: s1_arr}),
        field_meta=(
            {cfg.out: s1_full.field_meta.get(cfg.out)} if cfg.out in s1_full.field_meta else {}
        ),
    )


def s1_pipeline(cfg: S1RecipeConfig, reference: PointsDataSource):
    """Curry :func:`build_s1` with a fixed reference profile."""

    def run(cfd_profile: PointsDataSource) -> PointsDataSource:
        return build_s1(cfd_profile, reference, cfg)

    return run
