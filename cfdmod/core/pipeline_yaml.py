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
]

import pathlib
from typing import Callable, Literal

from pydantic import BaseModel, ConfigDict, Field

from cfdmod.core.data_source import DataSource
from cfdmod.core.protocols import Storage
from cfdmod.utils import read_yaml

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
    """Register an op under ``kind``.

    Idempotent: re-registering the same kind replaces the entry. Used
    to plug in extension ops without monkey-patching the registry
    constant.
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
        FieldSeriesForGroupsParams,
        FilterByGroupingParams,
        ProbeExtractionParams,
        ProfileInterpolationParams,
        StatisticsParams,
        compute_statistics,
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
        DivParams,
        ForceContributionParams,
        MomentContributionParams,
        MovingAverageParams,
        MulParams,
        ScaleParams,
        SubParams,
        add,
        div,
        force_contribution,
        moment_contribution,
        moving_average,
        mul,
        scale,
        sub,
    )
    from cfdmod.core.ops.geometric import (
        AttachGroupingParams,
        BodyGroupingParams,
        MeshAttachParams,
        RegroupTopologyParams,
        ZoningGroupingParams,
        attach_grouping,
        body_grouping,
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
        ("scale", scale, ScaleParams),
        ("attach_grouping", attach_grouping, AttachGroupingParams),
        ("mesh_attach", mesh_attach, MeshAttachParams),
        ("body_grouping", body_grouping, BodyGroupingParams),
        ("zoning_grouping", zoning_grouping, ZoningGroupingParams),
        ("regroup_topology", regroup_topology, RegroupTopologyParams),
        ("force_contribution", force_contribution, ForceContributionParams),
        ("moment_contribution", moment_contribution, MomentContributionParams),
        ("filter_by_grouping", filter_by_grouping, FilterByGroupingParams),
        ("field_series_for_groups", field_series_for_groups, FieldSeriesForGroupsParams),
        ("statistics", compute_statistics, StatisticsParams),
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
# Schema models
# ---------------------------------------------------------------------------


InputKind = Literal["surface", "volume", "points", "groups", "modes"]


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
        extras: Free-form fields forwarded to the storage adapter
            (e.g. ``group`` name for the H5 timeseries layout).
    """

    model_config = ConfigDict(extra="allow")

    source: str
    path: str
    format: Literal["xdmf_h5"] = "xdmf_h5"


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


def validate_template(template: PipelineTemplate) -> None:
    """Statically validate a template before any I/O.

    Walks the step DAG and raises on the errors a user is most likely to
    hit: unknown op kinds, dangling ``source`` / ``rhs`` references,
    duplicate step ids (or an id colliding with an input name), a ``rhs``
    on a unary op, and per-step params errors (missing required fields,
    typo'd fields caught by ``extra="forbid"``). Called by
    :func:`load_template`; also usable standalone on a programmatically
    built template.
    """
    _populate_default_registry()

    known: set[str] = set(template.inputs)
    for i, step in enumerate(template.pipeline):
        step_id = step.id or f"step_{i}"
        if step.kind not in OP_REGISTRY:
            raise KeyError(
                f"unknown op kind {step.kind!r} at step {step_id!r}; "
                f"registered kinds: {sorted(OP_REGISTRY)}"
            )
        arity, _, params_cls = OP_REGISTRY[step.kind]
        if step.source not in known:
            raise KeyError(
                f"step {step_id!r} references unknown source {step.source!r}; "
                f"known so far: {sorted(known)}"
            )
        if arity == "binary":
            if step.rhs is None:
                raise ValueError(f"step {step_id!r} is binary ({step.kind!r}) but has no rhs")
            if step.rhs not in known:
                raise KeyError(f"step {step_id!r} references unknown rhs {step.rhs!r}")
        elif step.rhs is not None:
            raise ValueError(
                f"step {step_id!r} is unary ({step.kind!r}) but has a rhs {step.rhs!r}; "
                "rhs is only valid on binary ops (add/sub/mul/div)"
            )
        # Build the params model so missing/typo'd fields fail here, not
        # after every input has already been read from disk.
        _step_params(step, params_cls, template.root)
        # Register the id last so a step cannot reference itself, and so a
        # duplicate id (or a clash with an input name) is caught.
        if step_id in known:
            raise ValueError(
                f"duplicate step id {step_id!r}; ids must be unique and must "
                "not collide with an input name"
            )
        known.add(step_id)

    for out_name, out in template.outputs.items():
        if out.source not in known:
            raise KeyError(
                f"output {out_name!r} references unknown source {out.source!r}; "
                f"known: {sorted(known)}"
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


def run_template(
    template: PipelineTemplate,
    *,
    storage: Storage,
) -> dict[str, DataSource]:
    """Run a parsed template against a :class:`Storage`.

    Returns the dict of all named values (inputs + step outputs) so
    callers can inspect intermediates. The ``outputs:`` block is
    written through ``storage.write_data_source`` as a side effect.
    """
    _populate_default_registry()
    # Static validation first: fail on typos/dangling refs before any I/O.
    validate_template(template)

    # 1. Load inputs.
    bindings: dict[str, DataSource] = {}
    for name, spec in template.inputs.items():
        # Storage keys are logical names. We treat the resolved path as
        # the storage key so the adapter can map it to its on-disk
        # layout.
        key = _resolve_key(template.root, spec.path)
        ds = storage.read_data_source(key)
        # Honor the declared kind: the H5 adapter infers surface-vs-points
        # from the filename, so a misnamed/misdeclared input would flow in
        # as the wrong kind silently. Assert the loaded kind matches.
        if ds.kind != spec.kind:
            raise ValueError(
                f"input {name!r} declares kind {spec.kind!r} but the source at "
                f"{spec.path!r} loaded as kind {ds.kind!r}"
            )
        bindings[name] = ds

    # 2. Walk pipeline.
    for i, step in enumerate(template.pipeline):
        step_id = step.id or f"step_{i}"
        if step.kind not in OP_REGISTRY:
            raise KeyError(
                f"unknown op kind {step.kind!r} at step {step_id!r}; "
                f"registered kinds: {sorted(OP_REGISTRY)}"
            )
        arity, fn, params_cls = OP_REGISTRY[step.kind]

        if step.source not in bindings:
            raise KeyError(f"step {step_id!r} references unknown source {step.source!r}")
        ds = bindings[step.source]
        params = _step_params(step, params_cls, template.root)

        if arity == "binary":
            if step.rhs is None:
                raise ValueError(f"step {step_id!r} is binary but has no rhs")
            if step.rhs not in bindings:
                raise KeyError(f"step {step_id!r} references unknown rhs {step.rhs!r}")
            result = fn(ds, bindings[step.rhs], params)
        else:
            result = fn(ds, params)

        bindings[step_id] = result

    # 3. Write outputs.
    for _, out in template.outputs.items():
        if out.source not in bindings:
            raise KeyError(f"output references unknown source {out.source!r}")
        key = _resolve_key(template.root, out.path)
        storage.write_data_source(key, bindings[out.source])

    return bindings
