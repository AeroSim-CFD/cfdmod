"""Field-level algebra dispatched on the four broadcasting rules.

Issue #131 explicitly calls out that recipes (Cp, Cf, S1, ...) should
not reimplement broadcasting. They route through this module instead.

The four rules from the proposal:

1. ``[multi or single] * constant`` -> uniform scaling.
2. ``[multi] * [single]`` with the same time axis -> column-wise
   (e.g. ``p - p_ref``: subtract a reference *per timestep*, broadcast
   across elements).
3. ``[multi] * [multi]`` with the same shape but only one carries
   timesteps -> row-wise (e.g. S1: divide a probe-time-series by a
   reference-time-aggregated profile, broadcast across timesteps).
4. ``[multi] * [multi]`` with the same shape -> element-wise.

Every public function takes :class:`DataSource` operands plus a target
field name, and returns a new :class:`DataSource` of the same kind as
``lhs`` with the result installed as a field. The :class:`FieldStore`
is asked for the underlying arrays once; we never assume the arrays are
loaded in RAM beyond what the store materialises.
"""

from __future__ import annotations

__all__ = [
    "BroadcastRule",
    "classify_broadcast",
    "add",
    "sub",
    "mul",
    "div",
]

from typing import Literal

import numpy as np

from cfdmod.core.data_source import DataSource
from cfdmod.core.field_meta import FieldMeta

BroadcastRule = Literal[
    "constant",
    "column",  # rule 2: same time axis, rhs has 1 element
    "row",  # rule 3: same n_elements, one side time-aggregated
    "elementwise",  # rule 4: same shape
]


def classify_broadcast(
    lhs_shape: tuple[int, ...],
    rhs_shape: tuple[int, ...] | None,
) -> BroadcastRule:
    """Classify which of the four rules applies.

    ``rhs_shape == None`` means the rhs is a Python scalar.

    Args:
        lhs_shape: Shape of the left operand field. Always
            ``(n_elements,)`` or ``(n_elements, n_timesteps)``.
        rhs_shape: Same convention for the right operand, or ``None``
            for a scalar.

    Raises:
        ValueError: If the shapes do not match any of the four rules.
    """
    if rhs_shape is None:
        return "constant"

    lhs_is_2d = len(lhs_shape) == 2
    rhs_is_2d = len(rhs_shape) == 2

    if lhs_shape == rhs_shape:
        # Covers both (n_elements,) vs (n_elements,) (time-aggregated
        # element-wise) and (n_elements, n_t) vs (n_elements, n_t).
        return "elementwise"

    if lhs_is_2d and rhs_is_2d and lhs_shape[1] == rhs_shape[1] and rhs_shape[0] == 1:
        return "column"

    # Row-wise: same n_elements, exactly one side time-aggregated (1-D).
    if (lhs_is_2d != rhs_is_2d) and lhs_shape[0] == rhs_shape[0]:
        return "row"

    raise ValueError(
        f"shapes {lhs_shape} and {rhs_shape} do not match any of the four "
        "broadcasting rules (constant, column, row, elementwise)"
    )


def _resolve_rhs(
    lhs: DataSource,
    rhs: DataSource | float | int,
    *,
    field: str,
) -> tuple[np.ndarray | None, BroadcastRule]:
    """Materialise the rhs array and pick the broadcast rule."""
    lhs_shape = lhs.fields.shape(field)
    if isinstance(rhs, (int, float, np.floating, np.integer)):
        return None, classify_broadcast(lhs_shape, None)
    if not isinstance(rhs, DataSource):
        raise TypeError(f"rhs must be a DataSource, int, or float; got {type(rhs).__name__}")
    rhs_shape = rhs.fields.shape(field)
    rule = classify_broadcast(lhs_shape, rhs_shape)
    rhs_arr = rhs.fields.read(field)
    return rhs_arr, rule


def _apply(
    lhs: DataSource,
    rhs: DataSource | float | int,
    *,
    field: str,
    out_field: str | None,
    op: callable,
) -> DataSource:
    """Run an algebra op against ``field`` and store under ``out_field``."""
    lhs_arr = lhs.fields.read(field)
    rhs_arr, rule = _resolve_rhs(lhs, rhs, field=field)

    if rule == "constant":
        result = op(lhs_arr, rhs)
    elif rule == "elementwise":
        result = op(lhs_arr, rhs_arr)
    elif rule == "column":
        # rhs is shape (1, n_timesteps); broadcast across elements axis.
        result = op(lhs_arr, rhs_arr)
    elif rule == "row":
        # one side is 1-D (n_elements,). NumPy broadcasting will lift it
        # along the time axis when both are paired in a 2-D op.
        if lhs_arr.ndim == 2 and rhs_arr.ndim == 1:
            result = op(lhs_arr, rhs_arr[:, None])
        else:
            result = op(lhs_arr[:, None], rhs_arr)
    else:  # pragma: no cover - exhaustive guard
        raise ValueError(f"unhandled broadcast rule {rule!r}")

    target = out_field or field
    meta = lhs.field_meta.get(target) or lhs.field_meta.get(field) or FieldMeta(name=target)
    return lhs.with_field(
        target, np.asarray(result), meta=meta.model_copy(update={"name": target})
    )


def add(
    lhs: DataSource,
    rhs: DataSource | float | int,
    *,
    field: str,
    out: str | None = None,
) -> DataSource:
    """Field-level addition, dispatched on the four broadcasting rules."""
    return _apply(lhs, rhs, field=field, out_field=out, op=np.add)


def sub(
    lhs: DataSource,
    rhs: DataSource | float | int,
    *,
    field: str,
    out: str | None = None,
) -> DataSource:
    """Field-level subtraction (e.g. ``p - p_ref`` for Cp)."""
    return _apply(lhs, rhs, field=field, out_field=out, op=np.subtract)


def mul(
    lhs: DataSource,
    rhs: DataSource | float | int,
    *,
    field: str,
    out: str | None = None,
) -> DataSource:
    """Field-level multiplication (e.g. scaling by ``1 / dyn_pressure``)."""
    return _apply(lhs, rhs, field=field, out_field=out, op=np.multiply)


def div(
    lhs: DataSource,
    rhs: DataSource | float | int,
    *,
    field: str,
    out: str | None = None,
) -> DataSource:
    """Field-level division (e.g. profile / reference for S1)."""
    return _apply(lhs, rhs, field=field, out_field=out, op=np.divide)
