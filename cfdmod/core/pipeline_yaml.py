"""YAML-as-Pipeline: load a v3 processing template and run it.

The schema is a flat list of *steps*. Each step has:

- ``id`` (optional): name by which downstream steps reference this
  step's output. Defaults to the step index as a string.
- ``kind``: the op kind (``sub``, ``moving_average``, ``statistics``,
  ...). Matches the registry in :data:`OP_REGISTRY`.
- ``source``: id of the data source the op consumes. May be an
  ``inputs:`` key on the first reference; thereafter it is the id of a
  previous step.
- ``rhs`` (binary ops only): id of the right-hand-side source.
- op-specific fields (``field``, ``out``, ``factor``, ``window``, ...)
  passed straight into the params model.

The runner is a small interpreter: it walks ``inputs`` -> loads via
the supplied :class:`Storage` -> walks ``pipeline`` -> dispatches each
step to the registered op -> records the output under the step id ->
walks ``outputs`` -> writes each named result via the same storage.

:func:`load_template` validates the whole template up front:
unknown op kinds, dangling ``source`` / ``rhs`` references, duplicate
step ids, ``rhs`` on a unary op, and per-step params (missing required
fields, typo'd fields) are all rejected before any input is read.

Example YAML::

    name: cp_default
    inputs:
      body:
        kind: surface
        path: body.h5
        field: pressure
      p_ref:
        kind: points
        path: probe.h5
        field: pressure
    pipeline:
      - id: cp_raw
        kind: sub
        source: body
        rhs: p_ref
        field: pressure
        out: cp
      - id: cp
        kind: scale
        source: cp_raw
        field: cp
        factor: 800.0
      - id: cp_stats
        kind: statistics
        source: cp
        field: cp
        kinds: [mean, rms, min, max]
    outputs:
      cp_timeseries:
        source: cp
        path: cp.time_series.h5
      cp_stats:
        source: cp_stats
        path: cp.stats.h5
"""

from __future__ import annotations

__all__ = [
    "InputSpec",
    "OutputSpec",
    "PipelineTemplate",
    "OP_REGISTRY",
    "register_op",
    "OpSpec",
    "BinaryOpSpec",
    "run_template",
    "load_template",
    "validate_template",
    "OpInfo",
    "list_ops",
    "op_info",
    "DigestStrategy",
    "FreshnessConfig",
]

import inspect
import pathlib
from typing import TYPE_CHECKING, Callable, Literal

from pydantic import BaseModel, ConfigDict, Field
from pydantic.json_schema import GenerateJsonSchema

from cfdmod.core.data_source import DataSource
from cfdmod.core.errors import (
    CfdmodError,
    OpError,
    TemplateError,
    TemplateReferenceError,
)
from cfdmod.core.protocols import Storage
from cfdmod.utils import read_yaml

if TYPE_CHECKING:
    from cfdmod.core.memory import ChunkPlan
    from cfdmod.core.progress import RunEvent

# ---------------------------------------------------------------------------
# Op registry
# ---------------------------------------------------------------------------

# An op entry is one of:
# - unary:  fn(ds, params) -> ds
# - binary: fn(ds, rhs, params) -> ds
#
# We tag each entry with its arity so the runner knows whether to read
# a ``rhs`` source. Params are constructed by Pydantic from the
# remaining step fields (everything except id/kind/source/rhs).

OpEntry = tuple[Literal["unary", "binary"], Callable[..., DataSource], type[BaseModel]]

OP_REGISTRY: dict[str, OpEntry] = {}


def register_op(
    kind: str,
    fn: Callable[..., DataSource],
    params_cls: type[BaseModel],
    *,
    arity: Literal["unary", "binary"] = "unary",
) -> None:
    """Register an op under ``kind`` -- the public extension point.

    A consumer adds a custom op by writing a function
    ``fn(ds, params) -> DataSource`` (or ``fn(ds, rhs, params)`` for a
    binary op) and a ``params_cls``, then calling this. The op is then a
    first-class citizen: it is usable in YAML/dict templates under its
    ``kind``, validated by :func:`validate_template`, and listed by
    :func:`list_ops`.

    For the op's data-source contract (``consumes`` / ``produces`` /
    ``requires_element_meta`` / ...) to be picked up by the catalog and
    the template linter, ``params_cls`` should subclass
    :class:`cfdmod.core.ops.OpParams` and set those class attributes; a
    plain ``BaseModel`` still registers but is treated as unconstrained.

    Idempotent: re-registering the same kind replaces the entry, so a
    consumer can also override a built-in.
    """
    OP_REGISTRY[kind] = (arity, fn, params_cls)


def _populate_default_registry() -> None:
    """Wire every built-in op into the registry.

    Kept as a function so the registry is populated lazily on first
    use; this avoids import cycles with the recipe layer.
    """
    if OP_REGISTRY:
        return

    from cfdmod.core.ops.data_source_create import (
        ExtremeValueParams,
        FaceCutParams,
        FieldSeriesForGroupsParams,
        FilterByGroupingParams,
        ProbeExtractionParams,
        ProfileInterpolationParams,
        StatisticsParams,
        compute_statistics,
        extreme_value,
        face_cut,
        field_series_for_groups,
        filter_by_grouping,
        probe_extraction,
        profile_interpolation,
    )
    from cfdmod.core.ops.data_source_create.modal_projection import (
        ModalProjectionParams,
        modal_projection,
    )
    from cfdmod.core.ops.data_source_create.modal_recomposition import (
        ModalRecompositionParams,
        modal_recomposition,
    )
    from cfdmod.core.ops.field import (
        AddParams,
        DerivativeParams,
        DivParams,
        ForceContributionParams,
        FrequencyFilterParams,
        MomentContributionParams,
        MovingAverageParams,
        MulParams,
        ScaleParams,
        SubParams,
        add,
        derivative,
        div,
        force_contribution,
        frequency_filter,
        moment_contribution,
        moving_average,
        mul,
        scale,
        sub,
    )
    from cfdmod.core.ops.geometric import (
        AttachGroupingParams,
        BodyGroupingParams,
        ConnectivityGroupingParams,
        MeshAttachParams,
        RegroupTopologyParams,
        ZoningGroupingParams,
        attach_grouping,
        body_grouping,
        connectivity_grouping,
        mesh_attach,
        regroup_topology,
        zoning_grouping,
    )
    from cfdmod.core.ops.time import (
        RescaleTimeParams,
        TranslateParams,
        WindowSelectionParams,
        rescale,
        translate,
        window_selection,
    )

    # Unary ops.
    for kind, fn, cls in [
        ("time_window", window_selection, WindowSelectionParams),
        ("time_translate", translate, TranslateParams),
        ("time_rescale", rescale, RescaleTimeParams),
        ("moving_average", moving_average, MovingAverageParams),
        ("derivative", derivative, DerivativeParams),
        ("frequency_filter", frequency_filter, FrequencyFilterParams),
        ("scale", scale, ScaleParams),
        ("attach_grouping", attach_grouping, AttachGroupingParams),
        ("mesh_attach", mesh_attach, MeshAttachParams),
        ("body_grouping", body_grouping, BodyGroupingParams),
        ("zoning_grouping", zoning_grouping, ZoningGroupingParams),
        ("connectivity_grouping", connectivity_grouping, ConnectivityGroupingParams),
        ("regroup_topology", regroup_topology, RegroupTopologyParams),
        ("force_contribution", force_contribution, ForceContributionParams),
        ("moment_contribution", moment_contribution, MomentContributionParams),
        ("filter_by_grouping", filter_by_grouping, FilterByGroupingParams),
        ("face_cut", face_cut, FaceCutParams),
        ("field_series_for_groups", field_series_for_groups, FieldSeriesForGroupsParams),
        ("statistics", compute_statistics, StatisticsParams),
        ("extreme_value", extreme_value, ExtremeValueParams),
        ("modal_projection", modal_projection, ModalProjectionParams),
        ("modal_recomposition", modal_recomposition, ModalRecompositionParams),
        ("probe_extraction", probe_extraction, ProbeExtractionParams),
        ("profile_interpolation", profile_interpolation, ProfileInterpolationParams),
    ]:
        register_op(kind, fn, cls, arity="unary")

    # Binary ops. The runner reads ``rhs`` from the step and passes the
    # resolved DataSource as the second positional argument.
    for kind, fn, cls in [
        ("add", add, AddParams),
        ("sub", sub, SubParams),
        ("mul", mul, MulParams),
        ("div", div, DivParams),
    ]:
        register_op(kind, fn, cls, arity="binary")


# ---------------------------------------------------------------------------
# Public op catalog (issue #147)
# ---------------------------------------------------------------------------

# The op registry is populated eagerly at import (bottom of this module), so a
# consumer can enumerate the op set without first running a template. The
# catalog below turns the registry into a stable, dependency-light description
# a node-based pipeline editor can consume: op kinds, arities, data-source
# contracts, and per-op parameter JSON Schemas.


class _LenientJsonSchema(GenerateJsonSchema):
    """JSON-schema generator that degrades gracefully on opaque types.

    Some op params carry numpy arrays or whole value objects (e.g.
    :class:`~cfdmod.core.grouping.Grouping`) that have no JSON-schema
    representation. Rather than fail the whole catalog, emit an empty
    (``{}`` = "any") schema for those fields; every scalar / string /
    enum field still renders normally for a form-building consumer.
    """

    def handle_invalid_for_json_schema(self, schema: object, error_info: str) -> dict:
        return {}


def _op_family(params_cls: type[BaseModel]) -> str:
    """Resolve the op family for a params class.

    An explicit ``op_family`` class attribute wins (custom ops set it);
    otherwise the family is inferred from the subpackage the op lives in,
    so built-in ops need no per-op bookkeeping. Families mirror
    :data:`cfdmod.core.ops.OpKind`.
    """
    declared = getattr(params_cls, "op_family", None)
    if declared:
        return declared
    mod = params_cls.__module__
    if ".ops.time." in mod:
        return "time"
    if ".ops.geometric." in mod:
        return "geometric"
    if ".ops.data_source_create." in mod:
        return "source_create"
    return "field"


class OpInfo(BaseModel):
    """Machine-readable description of one registered op.

    This is the unit returned by :func:`list_ops` / :func:`op_info`. It
    carries everything a consumer needs to render an op and validate a
    graph statically: the op ``kind`` (the string written under a step's
    ``kind:`` in a template), its ``arity``, its data-source contract, and
    the JSON Schema of its parameters.
    """

    kind: str
    family: str
    arity: Literal["unary", "binary"]
    consumes: list[str] | None
    produces: str
    requires_element_meta: list[str]
    produces_element_meta: list[str]
    replaces_fields: bool
    params_schema: dict


def _op_info(kind: str, entry: OpEntry) -> OpInfo:
    arity, _, params_cls = entry
    consumes = getattr(params_cls, "consumes", None)
    return OpInfo(
        kind=kind,
        family=_op_family(params_cls),
        arity=arity,
        consumes=None if consumes is None else sorted(consumes),
        produces=getattr(params_cls, "produces", "same"),
        requires_element_meta=sorted(getattr(params_cls, "requires_element_meta", frozenset())),
        produces_element_meta=sorted(getattr(params_cls, "produces_element_meta", frozenset())),
        replaces_fields=bool(getattr(params_cls, "replaces_fields", False)),
        params_schema=params_cls.model_json_schema(schema_generator=_LenientJsonSchema),
    )


def list_ops() -> list[OpInfo]:
    """Return the full op catalog, sorted by kind.

    Enumerates every registered op (built-ins plus any registered via
    :func:`register_op`) with its contract and parameter schema. Populates
    the registry on first call if it has not been already.
    """
    _populate_default_registry()
    return [_op_info(kind, OP_REGISTRY[kind]) for kind in sorted(OP_REGISTRY)]


def op_info(kind: str) -> OpInfo:
    """Return the :class:`OpInfo` for a single op kind.

    Raises ``KeyError`` if the kind is not registered.
    """
    _populate_default_registry()
    if kind not in OP_REGISTRY:
        raise KeyError(f"unknown op kind {kind!r}; registered kinds: {sorted(OP_REGISTRY)}")
    return _op_info(kind, OP_REGISTRY[kind])


# ---------------------------------------------------------------------------
# Schema models
# ---------------------------------------------------------------------------


InputKind = Literal["surface", "volume", "points", "groups", "modes"]

DigestStrategy = Literal["size_mtime", "content", "backend"]
"""How an input's change-detection token is derived. See :meth:`Storage.digest`."""


class FreshnessConfig(BaseModel):
    """Output-staleness settings for a template.

    Attributes:
        digest: Default strategy used to digest input dependencies when
            computing an output's signature. ``size_mtime`` (the default)
            reads no bytes.
        per_input: Optional per-input override, mapping an ``inputs:`` name
            to a strategy that wins over ``digest`` for that input only.
    """

    model_config = ConfigDict(extra="forbid")

    digest: DigestStrategy = "size_mtime"
    per_input: dict[str, DigestStrategy] = Field(default_factory=dict)


class InputSpec(BaseModel):
    """One entry under ``inputs:``.

    Attributes:
        kind: The :class:`~cfdmod.core.data_source.DataSource` kind this
            input is expected to be. ``run_template`` reads the source
            and asserts the loaded kind matches, so a mismatch (e.g. a
            probe file not named ``points.*``, which the H5 adapter would
            otherwise read as a surface) is caught rather than silently
            wrong.
        path: Absolute or repo-relative path to the input. Resolved
            against the template's ``root`` (see :func:`load_template`).
        field: For inputs that bundle a single field (probe / inflow),
            the field name on disk. Optional for multi-field inputs.
        extras: Free-form fields forwarded to the storage adapter
            (e.g. ``group`` selector for h5 timeseries).
    """

    model_config = ConfigDict(extra="allow")

    kind: InputKind
    path: str
    field: str | None = None


class OutputSpec(BaseModel):
    """One entry under ``outputs:``.

    Attributes:
        source: id of the step (or input) whose output is written.
        path: Destination path, resolved against the template root.
        format: Storage format tag. Only ``xdmf_h5`` is currently
            supported (the sole built-in :class:`Storage`).
        persist: Whether to write this output through the storage.
            ``False`` computes it and hands it back without touching disk --
            what a service wants when it will serialise the result itself.
        hold: Whether to keep this output in the dict ``run_template``
            returns. ``False`` lets its arrays be released as soon as it has
            been written, which is what a batch job wants.
        extras: Free-form fields forwarded to the storage adapter
            (e.g. ``group`` name for the H5 timeseries layout).

    Setting both to ``False`` would compute an output and throw it away, so it
    is rejected at validation.
    """

    model_config = ConfigDict(extra="allow")

    source: str
    path: str
    format: Literal["xdmf_h5"] = "xdmf_h5"
    persist: bool = True
    hold: bool = True


class OpSpec(BaseModel):
    """One pipeline step. Accepts arbitrary op-specific fields."""

    model_config = ConfigDict(extra="allow")

    id: str | None = None
    kind: str
    source: str
    rhs: str | None = None


class PipelineTemplate(BaseModel):
    """A complete YAML template."""

    model_config = ConfigDict(extra="forbid")

    name: str = "pipeline"
    root: str | None = None
    inputs: dict[str, InputSpec] = Field(default_factory=dict)
    pipeline: list[OpSpec] = Field(default_factory=list)
    outputs: dict[str, OutputSpec] = Field(default_factory=dict)
    freshness: FreshnessConfig = Field(default_factory=FreshnessConfig)


# Backwards-compat alias for symmetry with OpSpec.
BinaryOpSpec = OpSpec


# ---------------------------------------------------------------------------
# Loader / runner
# ---------------------------------------------------------------------------


def load_template(path: pathlib.Path | str) -> PipelineTemplate:
    """Load a YAML template from disk.

    ``root`` defaults to the directory containing the YAML file so
    relative ``path:`` entries inside ``inputs:`` / ``outputs:`` are
    resolved against the template's own location, not the caller's
    cwd.
    """
    p = pathlib.Path(path).resolve()
    data = read_yaml(p)
    if "root" not in data:
        data["root"] = str(p.parent)
    template = PipelineTemplate.model_validate(data)
    validate_template(template)
    return template


# Points sources carry coordinates intrinsically, so ``position`` element
# metadata is treated as available on any points binding even before an op
# populates ElementMeta.position explicitly.
_INTRINSIC_META = {"points": frozenset({"position"})}


class _BindingState:
    """Symbolic description of a binding tracked during static validation.

    Carries the data-source ``kind``, the set of available field names
    (``None`` = "unknown", i.e. not declared -> field checks are skipped
    to avoid false positives), and the set of available element-metadata
    keys.
    """

    __slots__ = ("kind", "fields", "meta")

    def __init__(self, kind: str, fields: frozenset[str] | None, meta: frozenset[str]) -> None:
        self.kind = kind
        self.fields = fields
        self.meta = meta


def _seed_meta(kind: str) -> frozenset[str]:
    return _INTRINSIC_META.get(kind, frozenset())


def _input_state(spec: "InputSpec") -> _BindingState:
    fields = frozenset({spec.field}) if spec.field else None
    return _BindingState(spec.kind, fields, _seed_meta(spec.kind))


def _consumed_fields(params: BaseModel) -> frozenset[str]:
    fn = getattr(params, "consumed_fields", None)
    return frozenset(fn()) if callable(fn) else frozenset()


def _produced_fields(params: BaseModel) -> frozenset[str]:
    fn = getattr(params, "produced_fields", None)
    return frozenset(fn()) if callable(fn) else frozenset()


def _next_state(
    params_cls: type[BaseModel], params: BaseModel, src: _BindingState
) -> _BindingState:
    """Compute the output binding state of an op applied to ``src``."""
    produces = getattr(params_cls, "produces", "same")
    produces_meta = frozenset(getattr(params_cls, "produces_element_meta", frozenset()))
    replaces = bool(getattr(params_cls, "replaces_fields", False))

    kind = src.kind if produces == "same" else produces
    if produces == "same":
        meta = src.meta | produces_meta
    else:
        # Fresh source: only the metadata the op sets, plus the new kind's
        # intrinsic metadata.
        meta = produces_meta | _seed_meta(kind)

    if replaces:
        fields: frozenset[str] | None = _produced_fields(params)
    elif src.fields is None:
        fields = None
    else:
        fields = src.fields | _produced_fields(params)
    return _BindingState(kind, fields, meta)


def _check_contract(
    step_id: str,
    step_kind: str,
    params_cls: type[BaseModel],
    params: BaseModel,
    src: _BindingState,
) -> None:
    """Validate one op against its source binding's kind / meta / fields.

    Strict on kind and element metadata (both deterministic); permissive
    on fields when the source's field set is unknown (undeclared input),
    so a valid template is never rejected for a field the linter merely
    could not see.
    """
    consumes = getattr(params_cls, "consumes", None)
    if consumes is not None and src.kind not in consumes:
        raise TemplateError(
            f"step {step_id!r} ({step_kind!r}) consumes a {sorted(consumes)} data source "
            f"but its source is kind {src.kind!r}"
        )

    missing_meta = frozenset(getattr(params_cls, "requires_element_meta", frozenset())) - src.meta
    if missing_meta:
        raise TemplateError(
            f"step {step_id!r} ({step_kind!r}) requires element metadata {sorted(missing_meta)} "
            f"not present on its source; attach it upstream (e.g. mesh_attach)"
        )

    if src.fields is not None:
        missing_fields = _consumed_fields(params) - src.fields
        if missing_fields:
            raise TemplateError(
                f"step {step_id!r} ({step_kind!r}) reads field(s) {sorted(missing_fields)} "
                f"not present on its source; available: {sorted(src.fields)}"
            )


def validate_template(template: PipelineTemplate) -> None:
    """Statically validate a template before any I/O.

    Walks the step DAG and raises on the errors a user is most likely to
    hit: unknown op kinds, dangling ``source`` / ``rhs`` references,
    duplicate step ids (or an id colliding with an input name), a ``rhs``
    on a unary op, and per-step params errors (missing required fields,
    typo'd fields caught by ``extra="forbid"``).

    It also runs a symbolic contract pass over the op catalog (issue
    #147): each step's declared ``consumes`` kind and ``requires_element_meta``
    are checked against the source binding, and field reads are checked when
    the field set is known. This catches graph-wiring mistakes -- e.g. a
    ``force_contribution`` before ``mesh_attach``, or a surface-only op on a
    points binding -- that a visual pipeline editor produces. The pass is
    strict on kind / metadata (deterministic) and permissive on fields when
    the source's fields were not declared. Called by :func:`load_template`;
    also usable standalone on a programmatically built template.
    """
    _populate_default_registry()

    known: set[str] = set(template.inputs)
    states: dict[str, _BindingState] = {
        name: _input_state(spec) for name, spec in template.inputs.items()
    }
    for i, step in enumerate(template.pipeline):
        step_id = step.id or f"step_{i}"
        if step.kind not in OP_REGISTRY:
            raise TemplateReferenceError(
                f"unknown op kind {step.kind!r} at step {step_id!r}; "
                f"registered kinds: {sorted(OP_REGISTRY)}"
            )
        arity, _, params_cls = OP_REGISTRY[step.kind]
        if step.source not in known:
            raise TemplateReferenceError(
                f"step {step_id!r} references unknown source {step.source!r}; "
                f"known so far: {sorted(known)}"
            )
        if arity == "binary":
            if step.rhs is None:
                raise TemplateError(f"step {step_id!r} is binary ({step.kind!r}) but has no rhs")
            if step.rhs not in known:
                raise TemplateReferenceError(
                    f"step {step_id!r} references unknown rhs {step.rhs!r}"
                )
        elif step.rhs is not None:
            raise TemplateError(
                f"step {step_id!r} is unary ({step.kind!r}) but has a rhs {step.rhs!r}; "
                "rhs is only valid on binary ops (add/sub/mul/div)"
            )
        # Build the params model so missing/typo'd fields fail here, not
        # after every input has already been read from disk.
        params = _step_params(step, params_cls, template.root)
        # Symbolic contract check + state propagation.
        src_state = states[step.source]
        _check_contract(step_id, step.kind, params_cls, params, src_state)
        # Register the id last so a step cannot reference itself, and so a
        # duplicate id (or a clash with an input name) is caught.
        if step_id in known:
            raise TemplateError(
                f"duplicate step id {step_id!r}; ids must be unique and must "
                "not collide with an input name"
            )
        known.add(step_id)
        states[step_id] = _next_state(params_cls, params, src_state)

    for out_name, out in template.outputs.items():
        if out.source not in known:
            raise TemplateReferenceError(
                f"output {out_name!r} references unknown source {out.source!r}; "
                f"known: {sorted(known)}"
            )
        if not out.persist and not out.hold:
            raise TemplateError(
                f"output {out_name!r} sets both persist and hold to false, so it would "
                "be computed and discarded; drop the output instead"
            )


def _resolve_key(template_root: str | None, path: str) -> str:
    """Resolve a template ``path:`` to the storage key.

    Storage adapters are keyed by stem (no extension): the H5 adapter
    resolves ``foo`` to ``<root>/foo.h5``. YAML templates may write
    ``path: foo``, ``path: foo.h5``, or an absolute path; we strip the
    ``.h5`` / ``.xdmf`` suffix uniformly so the storage sees a stem.

    The resolved key is anchored on the template's ``root:`` when the
    YAML path is relative; absolute paths and ``MemoryStorage`` keys
    (any string) pass through unchanged.
    """
    pp = pathlib.Path(path)
    if pp.suffix in {".h5", ".xdmf"}:
        pp = pp.with_suffix("")
    if pp.is_absolute() or template_root is None:
        return str(pp)
    return str(pathlib.Path(template_root) / pp)


# Step-level fields whose values are paths the user wrote relative to
# the template's root. The runner resolves them to absolute paths before
# building the op's params model so ops never need to know about the
# YAML's location.
_PATHLIKE_FIELDS = frozenset({"mesh", "mesh_path", "lnas", "csv"})


def _resolve_pathlike(value: object, template_root: str | None) -> object:
    if not isinstance(value, str) or template_root is None:
        return value
    pp = pathlib.Path(value)
    if pp.is_absolute():
        return value
    return str(pathlib.Path(template_root) / pp)


def _step_params(
    step: OpSpec,
    params_cls: type[BaseModel],
    template_root: str | None,
) -> BaseModel:
    """Build the params model from the step's extras.

    String fields whose name is in :data:`_PATHLIKE_FIELDS` are
    resolved against ``template_root`` so users can write relative
    paths in YAML.
    """
    raw = step.model_dump()
    for key in ("id", "kind", "source", "rhs"):
        raw.pop(key, None)
    for key, value in list(raw.items()):
        if key in _PATHLIKE_FIELDS:
            raw[key] = _resolve_pathlike(value, template_root)
    return params_cls.model_validate(raw)


def _accepts_kind(storage: Storage) -> bool:
    """Whether ``storage.read_data_source`` takes the ``kind`` keyword.

    ``Storage`` is a structural protocol, so a consumer's adapter written
    against the older two-argument signature is still a valid ``Storage``.
    Probing the signature once keeps those working, and -- unlike catching
    ``TypeError`` around the call -- cannot mistake a genuine ``TypeError``
    raised *inside* the adapter for an old signature.
    """
    try:
        params = inspect.signature(storage.read_data_source).parameters
    except (TypeError, ValueError):  # builtins / C extensions have no signature
        return False
    if "kind" in params:
        return True
    return any(p.kind is inspect.Parameter.VAR_KEYWORD for p in params.values())


def _live_time_arrays(template: PipelineTemplate) -> int:
    """How many time-resolved arrays the pipeline holds at its widest point.

    Used to price a time window. Counting exactly is not possible -- numpy
    temporaries inside an op are invisible from here -- so this is a floor:
    every declared input that carries a time axis, plus every step that
    produces one. That is conservative in the right direction (it over-counts
    live arrays, so it under-sizes the window) as long as ops do not allocate
    more than one extra array of their own, which the field ops do not.

    Never returns less than 1, so the caller can divide by it.
    """
    _populate_default_registry()
    # Inputs are assumed time-resolved: an aggregated one costs a single
    # column regardless of window size, so counting it is the safe error.
    live = len(template.inputs) + len(template.pipeline)
    return max(1, live)


def _chunkable_step_params(template: PipelineTemplate) -> list[BaseModel]:
    """Bound params for every step, for :func:`assert_time_chunkable`."""
    out: list[BaseModel] = []
    for step in template.pipeline:
        entry = OP_REGISTRY.get(step.kind)
        if entry is None:
            continue
        out.append(_step_params(step, entry[2], template.root))
    return out


def _plan_for(
    template: PipelineTemplate,
    bindings: dict[str, DataSource],
    chunk_size: int | None,
    memory_budget: int | None,
    n_live_arrays: int | None,
) -> "ChunkPlan":
    """Size the time window for this run from the loaded inputs.

    The shape comes from the widest time-resolved input: that is what a window
    of the pipeline actually costs. With no time-resolved input, or a single
    timestep, the plan is a single pass -- there is nothing to split.
    """
    from cfdmod.core.memory import plan_chunking

    resolved_live = n_live_arrays if n_live_arrays is not None else _live_time_arrays(template)
    timed = [ds for ds in bindings.values() if not ds.time.is_time_aggregated]
    n_timesteps = max((ds.time.n_timesteps for ds in timed), default=0)
    n_elements = max((ds.n_elements for ds in timed), default=0)

    if n_timesteps <= 1:
        return plan_chunking(n_elements, n_timesteps, n_live_arrays=resolved_live)
    return plan_chunking(
        n_elements,
        n_timesteps,
        budget_bytes=memory_budget,
        chunk_size=chunk_size,
        n_live_arrays=resolved_live,
    )


def _last_use(template: PipelineTemplate) -> dict[str, int]:
    """Step index after which each binding is no longer read.

    A binding is live until the last step that names it as ``source`` or
    ``rhs``; anything an output depends on is live to the end (represented by
    an index past the last step). Used to drop the runner's reference to an
    intermediate as soon as nothing downstream can ask for it -- the arrays
    then go when Python's refcount hits zero, which is the only safe way to
    release them: bindings share field arrays by reference, so the runner must
    never reach in and mutate a store.
    """
    end = len(template.pipeline)
    last: dict[str, int] = {}
    for i, step in enumerate(template.pipeline):
        for ref in (step.source, step.rhs):
            if ref:
                last[ref] = i
        last[step.id or f"step_{i}"] = last.get(step.id or f"step_{i}", i)
    for out in template.outputs.values():
        last[out.source] = end
    return last


def _retained_bindings(template: PipelineTemplate) -> set[str] | None:
    """Step ids whose per-window results must be kept, or ``None`` for all.

    Only what the ``outputs:`` block asks for survives a chunked run. That is
    not a convenience -- it is what makes chunking reduce anything. Keeping
    every intermediate for every window holds the full-size arrays *plus* the
    windowed copies, which costs strictly more than not chunking at all
    (measured: 25.8 MB against 11.4 MB unchunked, on a template whose only
    real output was a 4-group reduction of 4000 triangles).

    With no declared outputs there is nothing to select on, so everything is
    kept and chunking bounds transient allocations only.
    """
    sources = {out.source for out in template.outputs.values()}
    return sources or None


class _Reporter:
    """Bundles the ``on_progress`` / ``cancel`` seams so the runner takes one
    argument instead of threading two optionals through four functions.

    Both are optional and the no-op case costs an attribute check, so the
    unobserved path is unchanged.
    """

    __slots__ = ("_on_progress", "_cancel")

    def __init__(self, on_progress, cancel) -> None:
        self._on_progress = on_progress
        self._cancel = cancel

    def emit(self, phase, name, index, total, **extra) -> None:
        if self._on_progress is None:
            return
        from cfdmod.core.progress import RunEvent

        self._on_progress(RunEvent(phase=phase, name=name, index=index, total=total, **extra))

    def check(self, phase, name) -> None:
        """Raise :class:`RunCancelled` if the caller asked to stop."""
        if self._cancel is not None and self._cancel():
            from cfdmod.core.progress import RunCancelled

            raise RunCancelled(phase, name)


_NULL_REPORTER = _Reporter(None, None)


def _walk_chunked(
    template: PipelineTemplate,
    bindings: dict[str, DataSource],
    needed_steps: set[str] | None,
    plan: "ChunkPlan",
    reporter: "_Reporter" = _NULL_REPORTER,
    last_use: dict[str, int] | None = None,
) -> dict[str, DataSource]:
    """Run the step walk once per time window and concatenate the results.

    This is :func:`cfdmod.core.chunked.chunk_map_time` generalised to a
    multi-input template: every time-resolved binding is sliced to the same
    window, the whole walk runs on the slice, and the per-window results are
    concatenated along time. Time-aggregated bindings (a static reference
    pressure, a mesh) pass through untouched.

    Only the bindings :func:`_retained_bindings` selects are accumulated
    across windows; the rest go out of scope with their window, which is the
    entire source of the memory saving. The returned dict therefore carries
    the inputs (unsliced, as loaded) plus the retained results -- an
    intermediate that no output depends on is not reconstructed.

    Safe only for a time-length-preserving pipeline -- the caller must have
    run :func:`~cfdmod.core.chunked.assert_time_chunkable` first. An op that
    reduces the time axis (``statistics``) does not declare time
    chunkability precisely because windowed statistics are not the statistics
    of the whole series.
    """
    from cfdmod.core.chunked import concat_time, slice_time, time_windows

    retain = _retained_bindings(template)
    windows = list(time_windows(plan.n_timesteps, plan.chunk_size))
    accumulated: dict[str, list[DataSource]] = {}
    for w, sl in enumerate(windows):
        # Poll per window: that is the unit of work that actually takes time,
        # so it is the granularity at which cancelling is useful.
        reporter.check("step", f"window {w + 1}")
        window = {
            name: ds if ds.time.is_time_aggregated else slice_time(ds, sl)
            for name, ds in bindings.items()
        }
        produced = _walk_steps(
            template,
            window,
            needed_steps,
            reporter,
            window_index=w,
            n_windows=len(windows),
            last_use=last_use,
        )
        for name, ds in produced.items():
            if retain is not None and name not in retain:
                continue
            accumulated.setdefault(name, []).append(ds)
        # Drop the window's own bindings before allocating the next one.
        del produced, window

    merged: dict[str, DataSource] = dict(bindings)
    for name, parts in accumulated.items():
        if len(parts) == 1 or parts[0].time.is_time_aggregated:
            merged[name] = parts[0]
        else:
            merged[name] = concat_time(parts)
    return merged


def run_template(
    template: PipelineTemplate,
    *,
    storage: Storage,
    skip_fresh: bool = False,
    chunk_size: int | None = None,
    memory_budget: int | None = None,
    n_live_arrays: int | None = None,
    on_plan: Callable[["ChunkPlan"], None] | None = None,
    on_progress: Callable[["RunEvent"], None] | None = None,
    cancel: Callable[[], bool] | None = None,
    return_all: bool = False,
) -> dict[str, DataSource]:
    """Run a parsed template against a :class:`Storage`.

    Returns the dict of all named values (inputs + step outputs) so
    callers can inspect intermediates. The ``outputs:`` block is
    written through ``storage.write_data_source`` as a side effect.

    Each written output is stamped with a freshness signature (via
    ``storage.write_signature``) when the backend supports it, so a later
    :func:`~cfdmod.core.freshness.output_status` / ``skip_fresh`` run can
    tell fresh outputs from stale ones.

    With ``skip_fresh=True`` the runner first asks which outputs are stale
    (``freshness.output_status``), then runs only the steps and loads only
    the inputs those stale outputs depend on -- fresh outputs are neither
    recomputed nor rewritten. If every declared output is already fresh the
    run is a no-op and returns an empty binding dict.

    Time chunking
    -------------
    With ``chunk_size`` or ``memory_budget`` the pipeline runs over contiguous
    windows of the time axis and the per-window results are concatenated, so
    peak memory is ``O(n_elements * chunk)`` rather than
    ``O(n_elements * n_timesteps)``. The numbers are unchanged; only the peak
    is.

    Two things are worth being clear about:

    - **It is not free on every template.** The win is real when the pipeline
      collapses the element axis before the concatenation (a per-triangle force
      summed to a per-floor coefficient): the big intermediates then live for
      one window at a time while the concatenated result stays small. A
      pipeline that keeps the full element axis still bounds its transient
      allocations, but its final output is the same size as the unchunked one.
    - **Not every pipeline may be chunked.** Every op must declare ``"time"``
      in ``chunkable_along``. ``statistics`` deliberately does not -- the
      statistics of a window are not the statistics of the series -- so a
      template containing it raises before any I/O, naming the offending ops,
      rather than producing plausible wrong numbers.

    Args:
        chunk_size: Timesteps per window. Mutually exclusive with
            ``memory_budget``.
        memory_budget: Bytes the run may spend on time-resolved arrays; the
            window size is derived from it (see :mod:`cfdmod.core.memory`).
            Mutually exclusive with ``chunk_size``.
        n_live_arrays: Override for how many time-resolved arrays the budget
            arithmetic assumes are live at once. Derived from the template
            when omitted.
        on_plan: Called with the :class:`~cfdmod.core.memory.ChunkPlan` before
            execution starts, whether or not chunking is on. Use it to log or
            surface what the run decided; ``ChunkPlan.describe()`` renders a
            line.
        on_progress: Called with a :class:`~cfdmod.core.progress.RunEvent` as
            each input is loaded, each step runs, and each output is written.
        cancel: Polled at those same boundaries; returning True raises
            :class:`~cfdmod.core.progress.RunCancelled`. cfdmod cannot
            interrupt a numpy call in flight, so a run stops at the next
            boundary -- but the check precedes every write, so a cancelled run
            never leaves a partially written output set.
        return_all: Keep every intermediate binding alive and return it. Off by
            default: the runner otherwise drops each binding once no remaining
            step or output reads it, so the peak of a run is its widest live
            set rather than the sum of everything it ever computed. Turn it on
            for notebook work where inspecting intermediates is the point.

    Returns:
        The inputs, plus the outputs declared with ``hold: true`` (the
        default), plus -- with ``return_all`` -- every intermediate.
    """
    _populate_default_registry()
    # Static validation first: fail on typos/dangling refs before any I/O.
    validate_template(template)

    strategy = template.freshness.digest
    supports_freshness = hasattr(storage, "write_signature") and hasattr(storage, "digest")

    stale_outputs: set[str] | None = None
    needed_steps: set[str] | None = None
    needed_inputs: set[str] | None = None
    if skip_fresh:
        if not supports_freshness:
            raise TemplateError(
                "skip_fresh=True requires a storage backend with digest/read_signature/"
                "write_signature; this backend has none"
            )
        from cfdmod.core.freshness import closure_for_outputs, output_status

        statuses = output_status(template, storage, strategy)
        stale_outputs = {n for n, s in statuses.items() if not s.is_fresh}
        if template.outputs and not stale_outputs:
            # Everything is up to date -- nothing to load, run, or write.
            return {}
        needed_steps, needed_inputs = closure_for_outputs(template, stale_outputs)

    # 1. Load inputs (all, or -- under skip_fresh -- only those the stale
    #    outputs depend on).
    reporter = _Reporter(on_progress, cancel)
    # With no declared outputs there is nothing to select on, so keep
    # everything -- that template is being run for its intermediates.
    last_use = None if (return_all or not template.outputs) else _last_use(template)
    accepts_kind = _accepts_kind(storage)
    bindings: dict[str, DataSource] = {}
    n_inputs = len(template.inputs)
    for i, (name, spec) in enumerate(template.inputs.items()):
        if needed_inputs is not None and name not in needed_inputs:
            continue
        reporter.check("load", name)
        reporter.emit("load", name, i, n_inputs)
        # Storage keys are logical names. We treat the resolved path as
        # the storage key so the adapter can map it to its on-disk
        # layout.
        key = _resolve_key(template.root, spec.path)
        # Pass the declared kind down rather than letting the adapter guess.
        # The h5 layout does not record it, so without this the backend falls
        # back to the filename stem and a probe file not named ``points.*``
        # loads as a surface.
        try:
            if accepts_kind:
                ds = storage.read_data_source(key, kind=spec.kind)
            else:
                ds = storage.read_data_source(key)
        except ValueError as exc:
            raise TemplateError(
                f"input {name!r} declares kind {spec.kind!r}, which the storage "
                f"backend cannot read from {spec.path!r}: {exc}"
            ) from exc
        # Invariant check. With the kind passed explicitly this should be
        # unreachable; it stays so a backend that ignores the keyword cannot
        # feed the pipeline the wrong kind silently.
        if ds.kind != spec.kind:
            raise TemplateError(
                f"input {name!r} declares kind {spec.kind!r} but the source at "
                f"{spec.path!r} loaded as kind {ds.kind!r}"
            )
        bindings[name] = ds

    # 2. Decide whether and how to chunk the time axis, and say so.
    plan = _plan_for(template, bindings, chunk_size, memory_budget, n_live_arrays)
    if on_plan is not None:
        on_plan(plan)
    if plan.is_chunked:
        # Fail before any work if an op in the chain cannot be windowed.
        from cfdmod.core.chunked import assert_time_chunkable

        try:
            assert_time_chunkable(_chunkable_step_params(template))
        except ValueError as exc:
            raise TemplateError(f"cannot run this template chunked over time: {exc}") from exc

    # 3. Walk pipeline, over the whole time axis or one window at a time.
    if plan.is_chunked:
        bindings = _walk_chunked(template, bindings, needed_steps, plan, reporter, last_use)
    else:
        bindings = _walk_steps(template, bindings, needed_steps, reporter, last_use=last_use)

    return _write_outputs(
        template,
        bindings,
        storage,
        stale_outputs,
        supports_freshness,
        strategy,
        reporter,
        return_all,
    )


def _walk_steps(
    template: PipelineTemplate,
    bindings: dict[str, DataSource],
    needed_steps: set[str] | None,
    reporter: "_Reporter" = _NULL_REPORTER,
    *,
    window_index: int | None = None,
    n_windows: int | None = None,
    last_use: dict[str, int] | None = None,
) -> dict[str, DataSource]:
    """Execute the template's steps against ``bindings``, returning them extended.

    Split out of :func:`run_template` so the chunked runner can call it once per
    time window with windowed inputs. ``bindings`` is not mutated.
    """
    bindings = dict(bindings)
    total = len(template.pipeline)
    for i, step in enumerate(template.pipeline):
        step_id = step.id or f"step_{i}"
        if needed_steps is not None and step_id not in needed_steps:
            continue
        reporter.check("step", step_id)
        reporter.emit(
            "step",
            step_id,
            i,
            total,
            op_kind=step.kind,
            window=window_index,
            n_windows=n_windows,
        )
        if step.kind not in OP_REGISTRY:
            raise TemplateReferenceError(
                f"unknown op kind {step.kind!r} at step {step_id!r}; "
                f"registered kinds: {sorted(OP_REGISTRY)}"
            )
        arity, fn, params_cls = OP_REGISTRY[step.kind]

        if step.source not in bindings:
            raise TemplateReferenceError(
                f"step {step_id!r} references unknown source {step.source!r}"
            )
        ds = bindings[step.source]
        params = _step_params(step, params_cls, template.root)

        if arity == "binary":
            if step.rhs is None:
                raise TemplateError(f"step {step_id!r} is binary but has no rhs")
            if step.rhs not in bindings:
                raise TemplateReferenceError(
                    f"step {step_id!r} references unknown rhs {step.rhs!r}"
                )

        # Execute the op. A failure inside the op is wrapped as OpError with
        # the failing step id / kind, so a consumer can map it precisely
        # (rather than string-matching a bare exception) -- but cfdmod's own
        # TemplateError / TemplateReferenceError pass through untouched.
        try:
            if arity == "binary":
                result = fn(ds, bindings[step.rhs], params)
            else:
                result = fn(ds, params)
        except CfdmodError:
            raise
        except Exception as exc:
            raise OpError(
                f"step {step_id!r} ({step.kind!r}) raised while executing: {exc}",
                step_id=step_id,
                op_kind=step.kind,
            ) from exc

        bindings[step_id] = result

        # Drop our reference to anything no downstream step or output reads.
        # Refcounting frees the arrays; we never mutate a store, because
        # bindings share field arrays and another binding may still hold one.
        if last_use is not None:
            for name in [n for n, last in last_use.items() if last <= i and n in bindings]:
                del bindings[name]

    return bindings


def _write_outputs(
    template: PipelineTemplate,
    bindings: dict[str, DataSource],
    storage: Storage,
    stale_outputs: set[str] | None,
    supports_freshness: bool,
    strategy: str,
    reporter: "_Reporter" = _NULL_REPORTER,
    return_all: bool = False,
) -> dict[str, DataSource]:
    """Write the ``outputs:`` block, stamping freshness where supported.

    Honours each output's ``persist`` / ``hold``: ``persist: false`` computes it
    without touching storage, ``hold: false`` drops it from the returned dict
    once written.

    Cancellation is polled before each write, so a cancelled run never leaves a
    partially written output set.
    """
    sign = None
    if supports_freshness and any(o.persist for o in template.outputs.values()):
        from cfdmod.core.freshness import signature as sign

    total = len(template.outputs)
    dropped: set[str] = set()
    for i, (out_name, out) in enumerate(template.outputs.items()):
        if stale_outputs is not None and out_name not in stale_outputs:
            continue
        if out.source not in bindings:
            raise TemplateReferenceError(f"output references unknown source {out.source!r}")
        if out.persist:
            reporter.check("write", out_name)
            reporter.emit("write", out_name, i, total)
            key = _resolve_key(template.root, out.path)
            storage.write_data_source(key, bindings[out.source])
            if sign is not None:
                storage.write_signature(key, sign(template, out_name, storage, strategy))
        if not out.hold:
            dropped.add(out.source)

    # An output source is only released if *no* output that holds shares it.
    if not return_all:
        kept = {o.source for o in template.outputs.values() if o.hold}
        for name in dropped - kept:
            bindings.pop(name, None)

    return bindings


# Populate the op registry at import so consumers can enumerate ops (via
# list_ops / op_info / OP_REGISTRY) without first running a template. Safe:
# no op module imports this module, so there is no cycle.
_populate_default_registry()
