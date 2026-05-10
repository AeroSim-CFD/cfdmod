"""Op protocol classes.

Phase 1 only declares the four op-shape callable types and a Pydantic
discriminated parameter base. Concrete ops land in Phase 3 and beyond:

- ``cfdmod.core.ops.time.{window, translate, rescale}``
- ``cfdmod.core.ops.geometric.{rigid_body, rescale, group_belonging}``
- ``cfdmod.core.ops.data_source_create.{statistics, ...}``
- ``cfdmod.core.ops.field.{moving_average, algebra, ...}``

Holding the protocols here lets the pipeline layer reason about op
families without importing concrete implementations.
"""

from __future__ import annotations

__all__ = [
    "OpKind",
    "OpParams",
    "TimeOp",
    "GeometricOp",
    "SourceCreateOp",
    "FieldOp",
]

from typing import Callable, ClassVar, Literal

from pydantic import BaseModel, ConfigDict

from cfdmod.core.data_source import DataSource

OpKind = Literal["time", "geometric", "source_create", "field"]


class OpParams(BaseModel):
    """Discriminated parameter base for every op.

    Concrete op params subclass this and pin a ``Literal`` ``kind``
    field. The ``chunkable_along`` class attribute declares which
    chunking axes the op supports; the pipeline runner validates
    compatibility before running any chunked execution.
    """

    model_config = ConfigDict(frozen=True)

    kind: OpKind

    chunkable_along: ClassVar[frozenset[str]] = frozenset()


# Op signatures. Each op is a single-arg callable on DataSource produced by
# binding params via functools.partial -- the recipe constructs the binding,
# the pipeline runs the resulting callable.

TimeOp = Callable[[DataSource], DataSource]
GeometricOp = Callable[[DataSource], DataSource]
SourceCreateOp = Callable[[DataSource], DataSource]
FieldOp = Callable[[DataSource], DataSource]
