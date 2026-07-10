"""Modal projection -- forces in physical space -> generalized loads.

Given a :class:`SurfaceDataSource` (or :class:`PointsDataSource`)
carrying a per-element load timeseries ``f`` and a mode-shape matrix
``phi`` of shape ``(n_elements, n_modes)``, the generalized load on
mode ``i`` at time ``t`` is::

    Q_i(t) = sum_e phi[e, i] * f[e, t]

Returns a :class:`ModesDataSource` whose element axis is the mode
index and whose field is the generalized load timeseries.

This is the generic single-field modal projection. The building
dynamic-response pipeline uses the richer
:func:`~cfdmod.core.ops.data_source_create.generalized_building_load.generalized_building_load`
op, which projects the per-floor Cf / Cm channels with a CM lever-arm
correction instead of a plain ``phi.T @ f``.
"""

from __future__ import annotations

__all__ = ["ModalProjectionParams", "modal_projection"]

from typing import Any, ClassVar, Literal

import numpy as np
from pydantic import ConfigDict

from cfdmod.adapters.memory import MemoryFieldStore
from cfdmod.core.data_source import DataSource, ModesDataSource
from cfdmod.core.field_meta import FieldMeta
from cfdmod.core.ops import OpParams
from cfdmod.core.topology import ElementMeta


class ModalProjectionParams(OpParams):
    """Parameters for :func:`modal_projection`.

    Attributes:
        mode_shapes: ``(n_elements, n_modes)`` array. Same row ordering
            as the data source's element axis.
        field: Field name to project. Defaults to ``"force"``.
        out: Output field name on the modes data source. Defaults to
            ``"q"`` (generalized displacement / load symbol).
        mode_labels: Optional list of mode labels (length ``n_modes``)
            stored under the modes source's ``elements.annotations``.
    """

    model_config = ConfigDict(frozen=True, arbitrary_types_allowed=True)

    kind: Literal["modal_projection"] = "modal_projection"
    mode_shapes: Any
    field: str = "force"
    out: str = "q"
    mode_labels: list[str] | None = None

    chunkable_along: ClassVar[frozenset[str]] = frozenset({"time"})
    produces: ClassVar[str] = "modes"
    replaces_fields: ClassVar[bool] = True


def modal_projection(ds: DataSource, p: ModalProjectionParams) -> ModesDataSource:
    phi = np.asarray(p.mode_shapes, dtype=np.float64)
    if phi.ndim != 2 or phi.shape[0] != ds.n_elements:
        raise ValueError(
            f"mode_shapes must have shape (n_elements, n_modes); got {phi.shape}, "
            f"n_elements={ds.n_elements}"
        )

    f = np.asarray(ds.fields.read(p.field), dtype=np.float64)
    if f.ndim != 2:
        raise ValueError(f"field {p.field!r} must be 2-D (n_elements, n_timesteps); got {f.shape}")

    # Q[i, t] = sum_e phi[e, i] * f[e, t]   -> (n_modes, n_timesteps)
    q = phi.T @ f

    annotations: dict[str, Any] = {}
    if p.mode_labels is not None:
        if len(p.mode_labels) != q.shape[0]:
            raise ValueError(f"mode_labels length {len(p.mode_labels)} != n_modes {q.shape[0]}")
        annotations["mode_labels"] = list(p.mode_labels)

    src_meta = ds.field_meta.get(p.field)
    out_meta = (
        FieldMeta(name=p.out, unit=src_meta.unit, scale=src_meta.scale)
        if src_meta is not None
        else FieldMeta(name=p.out)
    )

    return ModesDataSource(
        time=ds.time,
        topology=None,
        elements=ElementMeta(annotations=annotations),
        fields=MemoryFieldStore({p.out: q}),
        field_meta={p.out: out_meta},
    )
