"""Generalized building load -- per-floor force/moment coefficients -> modal loads.

This is the building-specific projection step of the dynamic-response
pipeline. Unlike the generic :func:`modal_projection` (a plain
``phi.T @ f`` over a single scalar field), the structural generalized
load combines three per-floor load channels against three mode-shape
channels, with a center-of-mass (CM) lever-arm correction:

    Q_m(t) = sum_floor [ cf_x * DX_m + cf_y * DY_m + cm_z_onCM * RZ_m ]

where the floor torsion is referred to the CM before projection::

    cm_z_onCM = cm_z - (XR * cf_y - YR * cf_x)

(``XR, YR`` are the CM offsets per floor; the cross product matches the
legacy ``series_cross_product(arm, vx, vy) = arm[0]*vy - arm[1]*vx``).

The mode shapes must be **mass-normalized** (unit generalized mass) so
the downstream SDOF solver, which carries no explicit mass term, is
consistent. Normalization is the responsibility of the input builder
(see ``cfdmod.dynamics.structural``), not this op.
"""

from __future__ import annotations

__all__ = ["GeneralizedBuildingLoadParams", "generalized_building_load"]

from typing import Any, ClassVar, Literal

import numpy as np
from pydantic import ConfigDict

from cfdmod.adapters.memory import MemoryFieldStore
from cfdmod.core.data_source import DataSource, ModesDataSource
from cfdmod.core.field_meta import FieldMeta
from cfdmod.core.ops import OpParams
from cfdmod.core.topology import ElementMeta


class GeneralizedBuildingLoadParams(OpParams):
    """Parameters for :func:`generalized_building_load`.

    Attributes:
        mode_shapes: ``(n_floors, n_modes, 3)`` array of per-floor
            ``[DX, DY, RZ]`` components for each mode. Must be
            mass-normalized.
        cm_positions: ``(n_floors, 2)`` CM offsets ``[XR, YR]`` per floor.
        field_x: Name of the along-X force-coefficient field. Default ``"cf_x"``.
        field_y: Name of the along-Y force-coefficient field. Default ``"cf_y"``.
        field_mz: Name of the torsion moment-coefficient field. Default ``"cm_z"``.
        out: Output field on the modes data source. Default ``"q"``.
        mode_labels: Optional labels (length ``n_modes``).
    """

    model_config = ConfigDict(frozen=True, arbitrary_types_allowed=True)

    kind: Literal["generalized_building_load"] = "generalized_building_load"
    mode_shapes: Any
    cm_positions: Any
    field_x: str = "cf_x"
    field_y: str = "cf_y"
    field_mz: str = "cm_z"
    out: str = "q"
    mode_labels: list[str] | None = None

    chunkable_along: ClassVar[frozenset[str]] = frozenset({"time"})
    produces: ClassVar[str] = "modes"
    replaces_fields: ClassVar[bool] = True


def generalized_building_load(ds: DataSource, p: GeneralizedBuildingLoadParams) -> ModesDataSource:
    phi = np.asarray(p.mode_shapes, dtype=np.float64)
    if phi.ndim != 3 or phi.shape[0] != ds.n_elements or phi.shape[2] != 3:
        raise ValueError(
            f"mode_shapes must have shape (n_floors, n_modes, 3); got {phi.shape}, "
            f"n_floors={ds.n_elements}"
        )
    cm = np.asarray(p.cm_positions, dtype=np.float64)
    if cm.shape != (ds.n_elements, 2):
        raise ValueError(
            f"cm_positions must have shape (n_floors, 2); got {cm.shape}, n_floors={ds.n_elements}"
        )

    cf_x = np.asarray(ds.fields.read(p.field_x), dtype=np.float64)
    cf_y = np.asarray(ds.fields.read(p.field_y), dtype=np.float64)
    cm_z = np.asarray(ds.fields.read(p.field_mz), dtype=np.float64)
    for name, arr in ((p.field_x, cf_x), (p.field_y, cf_y), (p.field_mz, cm_z)):
        if arr.ndim != 2 or arr.shape[0] != ds.n_elements:
            raise ValueError(
                f"field {name!r} must be 2-D (n_floors, n_t) with n_floors={ds.n_elements}; "
                f"got {arr.shape}"
            )

    # Refer torsion to the CM: cm_z_onCM = cm_z - (XR*cf_y - YR*cf_x)
    xr = cm[:, 0][:, None]
    yr = cm[:, 1][:, None]
    cm_z_on_cm = cm_z - (xr * cf_y - yr * cf_x)

    dx = phi[:, :, 0]  # (n_floors, n_modes)
    dy = phi[:, :, 1]
    rz = phi[:, :, 2]

    # Q[m, t] = sum_floor DX[f,m]*cf_x[f,t] + DY[f,m]*cf_y[f,t] + RZ[f,m]*cm_z_onCM[f,t]
    q = dx.T @ cf_x + dy.T @ cf_y + rz.T @ cm_z_on_cm

    annotations: dict[str, Any] = {}
    if p.mode_labels is not None:
        if len(p.mode_labels) != q.shape[0]:
            raise ValueError(f"mode_labels length {len(p.mode_labels)} != n_modes {q.shape[0]}")
        annotations["mode_labels"] = list(p.mode_labels)

    return ModesDataSource(
        time=ds.time,
        topology=None,
        elements=ElementMeta(annotations=annotations),
        fields=MemoryFieldStore({p.out: q}),
        field_meta={p.out: FieldMeta(name=p.out)},
    )
