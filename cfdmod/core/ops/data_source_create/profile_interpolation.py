"""Vertical-profile reinterpolation -- 1-D linear over the z axis.

Used by the S1 recipe: a CFD profile sampled at one set of heights
needs to be lifted onto the heights of a reference profile so the two
can be divided element-wise.

Operates exclusively on :class:`PointsDataSource` whose
``elements.position`` z column gives the heights. Time-resolved fields
are interpolated per timestep; time-aggregated fields are interpolated
in 1-D.
"""

from __future__ import annotations

__all__ = ["ProfileInterpolationParams", "profile_interpolation"]

from typing import Any, ClassVar, Literal

import numpy as np
from pydantic import ConfigDict

from cfdmod.adapters.memory import MemoryFieldStore
from cfdmod.core.data_source import PointsDataSource
from cfdmod.core.ops import OpParams
from cfdmod.core.topology import ElementMeta, Topology


class ProfileInterpolationParams(OpParams):
    """Parameters for :func:`profile_interpolation`.

    Attributes:
        target_heights: 1-D array of new heights.
        field: Field to reinterpolate. Defaults to ``"u"``.
    """

    model_config = ConfigDict(frozen=True, arbitrary_types_allowed=True)

    kind: Literal["profile_interpolation"] = "profile_interpolation"
    target_heights: Any
    field: str = "u"

    chunkable_along: ClassVar[frozenset[str]] = frozenset({"time"})
    consumes: ClassVar[frozenset[str] | None] = frozenset({"points"})
    produces: ClassVar[str] = "points"
    replaces_fields: ClassVar[bool] = True


def profile_interpolation(ds: PointsDataSource, p: ProfileInterpolationParams) -> PointsDataSource:
    if ds.elements.position is None:
        raise ValueError("profile_interpolation requires elements.position on the source")
    z_src = ds.elements.position[:, 2]
    z_target = np.asarray(p.target_heights, dtype=np.float64).ravel()

    order = np.argsort(z_src)
    z_sorted = z_src[order]

    arr = np.asarray(ds.fields.read(p.field), dtype=np.float64)
    is_time = arr.ndim == 2
    if is_time:
        out = np.empty((z_target.size, arr.shape[1]), dtype=np.float64)
        for t in range(arr.shape[1]):
            out[:, t] = np.interp(z_target, z_sorted, arr[order, t])
    else:
        out = np.interp(z_target, z_sorted, arr[order])

    new_pos = np.zeros((z_target.size, 3), dtype=np.float64)
    new_pos[:, 2] = z_target

    return PointsDataSource(
        time=ds.time,
        topology=Topology.points(new_pos),
        elements=ElementMeta(position=new_pos),
        fields=MemoryFieldStore({p.field: out}),
        field_meta={p.field: ds.field_meta.get(p.field)} if p.field in ds.field_meta else {},
    )
