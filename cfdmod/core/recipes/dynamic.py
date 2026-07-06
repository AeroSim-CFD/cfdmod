"""Dynamic-analysis recipe -- per-element loads -> modal solution -> physical response.

Per the odt::

    container of Cf + modes data -> solution as modal displacements
    container of solutions       -> displacement / acceleration / loads
                                    in original coordinates

This recipe is the small-data analogue of the legacy
``cfdmod.hfpi.dynamic`` pipeline. It wires three of the Phase 6
primitives together:

1. :func:`modal_projection` -- physical-space load timeseries ``f``
   times mode shapes ``phi`` -> generalized loads ``Q``.
2. A user-supplied modal solver (``Q -> q``) -- the linear ODE
   ``Mq'' + Cq' + Kq = Q`` is the SDOF case per mode and lives outside
   the algebra layer. We accept any callable that maps a
   :class:`ModesDataSource` to another :class:`ModesDataSource`.
3. :func:`modal_recomposition` -- modal coordinates back into the
   physical mesh.

A trivial ``identity`` solver is provided for tests and for the
"already solved" path (Q == q, e.g. quasi-static modal scaling).
"""

from __future__ import annotations

__all__ = [
    "DynamicAnalysisConfig",
    "build_dynamic_response",
    "identity_solver",
]

from typing import Any, Callable

import numpy as np
from pydantic import BaseModel, ConfigDict

from cfdmod.core.data_source import DataSource, ModesDataSource, PointsDataSource
from cfdmod.core.ops.data_source_create.modal_projection import (
    ModalProjectionParams,
    modal_projection,
)
from cfdmod.core.ops.data_source_create.modal_recomposition import (
    ModalRecompositionParams,
    modal_recomposition,
)

ModalSolver = Callable[[ModesDataSource], ModesDataSource]


def identity_solver(modes: ModesDataSource) -> ModesDataSource:
    """Pass-through solver. Useful when the caller already has the modal
    response (e.g. quasi-static scaling) or for tests."""
    return modes


class DynamicAnalysisConfig(BaseModel):
    """Dynamic-analysis recipe parameters.

    Attributes:
        mode_shapes: ``(n_load_elements, n_modes)`` mode-shape matrix at
            the load points (used to compute ``Q``). For most cases the
            same mode shapes also drive recomposition (see
            ``recomposition_mode_shapes``).
        recomposition_mode_shapes: Optional ``(n_target_elements,
            n_modes)`` matrix evaluated at the *target* coordinates. If
            ``None``, ``mode_shapes`` is reused (load and target
            coincide).
        target_points: ``(n_target_elements, 3)`` coordinates for the
            recomposed response.
        load_field: Field name carrying the load timeseries on the
            input data source. Defaults to ``"force"``.
        response_field: Field name on the output points data source.
            Defaults to ``"u"``.
    """

    model_config = ConfigDict(frozen=True, arbitrary_types_allowed=True)

    mode_shapes: Any
    target_points: Any
    recomposition_mode_shapes: Any | None = None
    load_field: str = "force"
    response_field: str = "u"


def build_dynamic_response(
    load_source: DataSource,
    cfg: DynamicAnalysisConfig,
    *,
    solver: ModalSolver = identity_solver,
) -> PointsDataSource:
    """Assemble the recipe end-to-end."""
    phi = np.asarray(cfg.mode_shapes, dtype=np.float64)
    modes = modal_projection(
        load_source,
        ModalProjectionParams(mode_shapes=phi, field=cfg.load_field, out="q"),
    )
    solved = solver(modes)
    phi_target = (
        phi
        if cfg.recomposition_mode_shapes is None
        else np.asarray(cfg.recomposition_mode_shapes, dtype=np.float64)
    )
    return modal_recomposition(
        solved,
        ModalRecompositionParams(
            mode_shapes=phi_target,
            target_points=np.asarray(cfg.target_points, dtype=np.float64),
            field="q",
            out=cfg.response_field,
        ),
    )
