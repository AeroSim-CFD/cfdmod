"""Time-derivative field op -- velocity / acceleration from a series.

Finite-difference derivative along the time axis. The op preserves the
element axis and the time axis (same ``n_timesteps``); it only rewrites
one named field, so it belongs to the field-op family alongside
:mod:`cfdmod.core.ops.field.moving_average`.

Two orders are supported, reproducing the legacy HFPI stencils
(``cfdmod.hfpi.common.first_derivative`` / ``second_derivative``) exactly:

- ``order=1`` (velocity): backward difference for interior points, a
  forward difference for the first point.
- ``order=2`` (acceleration): central difference for interior points,
  one-sided three-point stencils at both boundaries.

v3 fields are stored ``(n_elements, n_timesteps)``; the derivative runs
along ``axis=1`` (time), unlike the legacy ``(n_timesteps, n_floors)``
layout which differentiated along ``axis=0``.
"""

from __future__ import annotations

__all__ = ["DerivativeParams", "derivative"]

from typing import ClassVar, Literal

import numpy as np

from cfdmod.core.data_source import DataSource
from cfdmod.core.ops import OpParams

_DEFAULT_OUT = {1: "velocity", 2: "acceleration"}


class DerivativeParams(OpParams):
    """Parameters for :func:`derivative`.

    Attributes:
        order: Derivative order. ``1`` for velocity, ``2`` for
            acceleration.
        field: Name of the field to differentiate. Defaults to
            ``"displacement"``.
        out: Optional output field name. Defaults to ``"velocity"`` for
            ``order=1`` and ``"acceleration"`` for ``order=2``.
    """

    kind: Literal["derivative"] = "derivative"
    order: Literal[1, 2]
    field: str = "displacement"
    out: str | None = None

    chunkable_along: ClassVar[frozenset[str]] = frozenset({"elements"})

    def produced_fields(self) -> frozenset[str]:
        return frozenset({self.out or _DEFAULT_OUT[self.order]})


def _first_derivative(arr: np.ndarray, dt: float) -> np.ndarray:
    """Backward difference interior, forward difference at index 0.

    ``arr`` shape ``(n_elements, n_timesteps)``; derivative along axis 1.
    """
    out = np.zeros_like(arr, dtype=np.float64)
    out[:, 1:] = (arr[:, 1:] - arr[:, :-1]) / dt
    out[:, 0] = (arr[:, 1] - arr[:, 0]) / dt
    return out


def _second_derivative(arr: np.ndarray, dt: float) -> np.ndarray:
    """Central difference interior, one-sided three-point at the edges."""
    out = np.zeros_like(arr, dtype=np.float64)
    out[:, 1:-1] = (arr[:, 2:] - 2 * arr[:, 1:-1] + arr[:, :-2]) / dt**2
    out[:, 0] = (arr[:, 2] - 2 * arr[:, 1] + arr[:, 0]) / dt**2
    out[:, -1] = (arr[:, -1] - 2 * arr[:, -2] + arr[:, -3]) / dt**2
    return out


def derivative(ds: DataSource, p: DerivativeParams) -> DataSource:
    if ds.time.is_time_aggregated:
        raise ValueError("derivative requires a time-resolved data source")
    min_steps = 2 if p.order == 1 else 3
    if ds.time.n_timesteps < min_steps:
        raise ValueError(
            f"derivative order={p.order} requires at least {min_steps} timesteps "
            f"(got n_timesteps={ds.time.n_timesteps})"
        )

    dt = float(ds.time.timestep_size)
    arr = np.asarray(ds.fields.read(p.field), dtype=np.float64)
    if arr.ndim != 2:
        raise ValueError(
            f"field {p.field!r} must be 2-D (n_elements, n_timesteps); got shape {arr.shape}"
        )

    out_arr = _first_derivative(arr, dt) if p.order == 1 else _second_derivative(arr, dt)

    target = p.out or _DEFAULT_OUT[p.order]
    return ds.with_field(target, out_arr)
