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

from cfdmod.core.data_source import DataSource, DataSourceKind

OpKind = Literal["time", "geometric", "source_create", "field"]


class OpParams(BaseModel):
    """Discriminated parameter base for every op.

    Concrete op params subclass this and pin a ``Literal`` ``kind``
    field. The ``chunkable_along`` class attribute declares which
    chunking axes the op supports; the pipeline runner validates
    compatibility before running any chunked execution.

    Service-contract metadata (issue #147)
    --------------------------------------
    The class attributes below declare the op's data-source contract so
    a consumer (e.g. a node-based pipeline editor) can validate a graph
    statically -- before any I/O -- and render the op without importing
    its implementation. They carry deliberately permissive defaults so a
    new op is unconstrained unless it opts in:

    - :attr:`consumes` -- the set of input :class:`DataSource` kinds the
      op accepts, or ``None`` for "no static constraint".
    - :attr:`produces` -- the output kind, or the sentinel ``"same"`` for
      kind-preserving ops.
    - :attr:`requires_element_meta` / :attr:`produces_element_meta` --
      per-element attributes (``area`` / ``normal`` / ``position``) the op
      needs on its input, and the ones it adds to its output.
    - :attr:`replaces_fields` -- ``True`` when the op emits a brand-new
      field set (statistics, per-group aggregations, probe extraction),
      ``False`` when it adds to / rewrites the incoming fields.

    :meth:`consumed_fields` / :meth:`produced_fields` derive the field
    names from the bound params (``field`` / ``out``); ops with a
    non-standard field shape override them.
    """

    # extra="forbid" so a typo'd step field in a YAML template (e.g.
    # ``windows:`` for ``window:``) is a hard error rather than being
    # silently dropped and the op running with its default -> wrong numbers.
    model_config = ConfigDict(frozen=True, extra="forbid")

    kind: OpKind

    chunkable_along: ClassVar[frozenset[str]] = frozenset()

    # --- service contract (issue #147) ---------------------------------
    # Op family (time/geometric/source_create/field). Built-in ops leave
    # this None and are classified by their module path; custom ops
    # defined outside cfdmod set it explicitly so the catalog reports the
    # right family.
    op_family: ClassVar[OpKind | None] = None
    consumes: ClassVar[frozenset[DataSourceKind] | None] = None
    produces: ClassVar[str] = "same"
    requires_element_meta: ClassVar[frozenset[str]] = frozenset()
    produces_element_meta: ClassVar[frozenset[str]] = frozenset()
    replaces_fields: ClassVar[bool] = False

    def consumed_fields(self) -> frozenset[str]:
        """Field names this op reads from its input, given these params.

        Default: the ``field`` attribute if present. Ops that read a set
        of fields (e.g. per-direction force components) override this.
        """
        f = getattr(self, "field", None)
        return frozenset({f}) if isinstance(f, str) and f else frozenset()

    def produced_fields(self) -> frozenset[str]:
        """Field names this op writes to its output, given these params.

        Default: the ``out`` alias if set, else the in-place ``field``.
        Ops that emit several fields (statistics, force/moment
        contributions) override this.
        """
        out = getattr(self, "out", None)
        if isinstance(out, str) and out:
            return frozenset({out})
        return self.consumed_fields()


# Op signatures. Each op is a single-arg callable on DataSource produced by
# binding params via functools.partial -- the recipe constructs the binding,
# the pipeline runs the resulting callable.

TimeOp = Callable[[DataSource], DataSource]
GeometricOp = Callable[[DataSource], DataSource]
SourceCreateOp = Callable[[DataSource], DataSource]
FieldOp = Callable[[DataSource], DataSource]
