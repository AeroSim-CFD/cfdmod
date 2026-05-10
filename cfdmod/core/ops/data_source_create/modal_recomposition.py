"""Modal recomposition -- modal coordinates -> physical-space response.

Inverse of :func:`modal_projection`. Given a :class:`ModesDataSource`
carrying generalized displacements ``q`` of shape ``(n_modes,
n_timesteps)`` and a mode-shape matrix ``phi`` of shape
``(n_target_elements, n_modes)``, the per-element physical response
is::

    u[e, t] = sum_i phi[e, i] * q[i, t]

The op needs a target topology / element metadata for the result
(typically the structural mesh). The caller supplies them as part of
:class:`ModalRecompositionParams`.
"""

from __future__ import annotations

__all__ = ["ModalRecompositionParams", "modal_recomposition"]

from typing import Any, ClassVar, Literal

import numpy as np
from pydantic import ConfigDict

from cfdmod.adapters.memory import MemoryFieldStore
from cfdmod.core.data_source import DataSource, ModesDataSource, PointsDataSource
from cfdmod.core.field_meta import FieldMeta
from cfdmod.core.ops import OpParams
from cfdmod.core.topology import ElementMeta, Topology


class ModalRecompositionParams(OpParams):
    """Parameters for :func:`modal_recomposition`.

    Attributes:
        mode_shapes: ``(n_target_elements, n_modes)`` matrix.
        target_points: ``(n_target_elements, 3)`` coordinates for the
            output points data source.
        field: Modal field on the input modes source. Defaults to
            ``"q"``.
        out: Output field name on the resulting points data source.
            Defaults to ``"u"`` (displacement).
    """

    model_config = ConfigDict(frozen=True, arbitrary_types_allowed=True)

    kind: Literal["modal_recomposition"] = "modal_recomposition"
    mode_shapes: Any
    target_points: Any
    field: str = "q"
    out: str = "u"

    chunkable_along: ClassVar[frozenset[str]] = frozenset({"time"})


def modal_recomposition(ds: ModesDataSource, p: ModalRecompositionParams) -> PointsDataSource:
    if not isinstance(ds, ModesDataSource):
        raise TypeError("modal_recomposition expects a ModesDataSource input")

    phi = np.asarray(p.mode_shapes, dtype=np.float64)
    pts = np.asarray(p.target_points, dtype=np.float64)
    if phi.ndim != 2 or phi.shape[0] != pts.shape[0]:
        raise ValueError(
            f"mode_shapes shape {phi.shape} incompatible with target_points "
            f"({pts.shape})"
        )

    q = np.asarray(ds.fields.read(p.field), dtype=np.float64)
    if q.ndim != 2 or q.shape[0] != phi.shape[1]:
        raise ValueError(
            f"modes field {p.field!r} shape {q.shape} incompatible with "
            f"mode_shapes second axis ({phi.shape[1]})"
        )

    u = phi @ q  # (n_target_elements, n_timesteps)

    src_meta = ds.field_meta.get(p.field)
    out_meta = (
        FieldMeta(name=p.out, unit=src_meta.unit, scale=src_meta.scale)
        if src_meta is not None
        else FieldMeta(name=p.out)
    )

    return PointsDataSource(
        time=ds.time,
        topology=Topology.points(pts),
        elements=ElementMeta(position=pts),
        fields=MemoryFieldStore({p.out: u}),
        field_meta={p.out: out_meta},
    )
