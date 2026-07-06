"""Algebra ops as :class:`OpParams` wrappers around :mod:`cfdmod.core.algebra`.

The four broadcasting rules from issue #131 are implemented once in
:mod:`cfdmod.core.algebra`. This module re-exposes them as
:class:`OpParams` plus single-argument callables suitable for binding
into a :class:`Pipeline` via :func:`functools.partial`.

Recipes use these wrappers when they want a pipeline step ("subtract a
reference field, then scale by a constant"). Recipes that just want to
combine two arrays in-place can call the underlying functions directly.
"""

from __future__ import annotations

__all__ = [
    "AddParams",
    "SubParams",
    "MulParams",
    "DivParams",
    "add",
    "sub",
    "mul",
    "div",
    "ScaleParams",
    "scale",
]

from typing import ClassVar, Literal

from cfdmod.core import algebra
from cfdmod.core.data_source import DataSource
from cfdmod.core.ops import OpParams

# Binary ops require an explicit rhs DataSource passed at recipe-build
# time (via partial). Hence the params model carries only the field name
# and an optional out alias; the rhs is bound by the recipe.


class AddParams(OpParams):
    kind: Literal["field_add"] = "field_add"
    field: str
    out: str | None = None
    chunkable_along: ClassVar[frozenset[str]] = frozenset({"time", "elements"})


class SubParams(OpParams):
    kind: Literal["field_sub"] = "field_sub"
    field: str
    out: str | None = None
    chunkable_along: ClassVar[frozenset[str]] = frozenset({"time", "elements"})


class MulParams(OpParams):
    kind: Literal["field_mul"] = "field_mul"
    field: str
    out: str | None = None
    chunkable_along: ClassVar[frozenset[str]] = frozenset({"time", "elements"})


class DivParams(OpParams):
    kind: Literal["field_div"] = "field_div"
    field: str
    out: str | None = None
    chunkable_along: ClassVar[frozenset[str]] = frozenset({"time", "elements"})


def add(ds: DataSource, rhs: DataSource | float | int, p: AddParams) -> DataSource:
    return algebra.add(ds, rhs, field=p.field, out=p.out)


def sub(ds: DataSource, rhs: DataSource | float | int, p: SubParams) -> DataSource:
    return algebra.sub(ds, rhs, field=p.field, out=p.out)


def mul(ds: DataSource, rhs: DataSource | float | int, p: MulParams) -> DataSource:
    return algebra.mul(ds, rhs, field=p.field, out=p.out)


def div(ds: DataSource, rhs: DataSource | float | int, p: DivParams) -> DataSource:
    return algebra.div(ds, rhs, field=p.field, out=p.out)


# Convenience: scaling by a constant is the most common single-operand
# field op (Cp = (p - p_ref) / dyn_pressure -> the second factor is the
# scale). Express it as a unary op that bakes in the scalar at recipe-
# construction time, so the resulting callable is a straight Pipeline step.


class ScaleParams(OpParams):
    """Scale ``field`` by a constant; uniform broadcast across all axes."""

    kind: Literal["field_scale"] = "field_scale"
    field: str
    factor: float
    out: str | None = None
    chunkable_along: ClassVar[frozenset[str]] = frozenset({"time", "elements"})


def scale(ds: DataSource, p: ScaleParams) -> DataSource:
    return algebra.mul(ds, p.factor, field=p.field, out=p.out)
