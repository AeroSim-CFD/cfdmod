"""Output-staleness detection for pipeline templates.

A generated output is *fresh* when nothing that determines it has changed
since it was last written, and *stale* otherwise. "Everything that
determines it" is captured by a **signature**: a hash over

1. the params + wiring of every step in the output's dependency subgraph,
2. a change-detecting digest of each ``inputs:`` file that subgraph
   touches (obtained via :meth:`Storage.digest` -- no byte transfer for the
   default ``size_mtime`` strategy, an object-store ETag for ``backend``),
3. a code/format version tag, so a change in op semantics or the on-disk
   layout invalidates old outputs.

:func:`run_template` stamps the signature next to each output it writes
(``Storage.write_signature``); :func:`output_status` recomputes and
compares, and ``run_template(..., skip_fresh=True)`` uses the comparison to
skip recomputing fresh outputs and prune the steps only they need.

The dependency walk mirrors the wiring the runner and
:func:`~cfdmod.core.pipeline_yaml.validate_template` already use: each step
depends on its ``source`` (plus ``rhs`` for a binary op, plus any
``sources`` map a recipe step declares). It reads no data -- only metadata
digests -- so freshness decisions are cheap.
"""

from __future__ import annotations

__all__ = [
    "FORMAT_VERSION",
    "DEFAULT_DIGEST",
    "OutputStatus",
    "Status",
    "code_version",
    "immediate_deps",
    "output_closure",
    "closure_for_outputs",
    "signature",
    "output_status",
    "resolve_strategy",
]

import hashlib
import json
from typing import Literal

from pydantic import BaseModel

from cfdmod.core.errors import StorageKeyError
from cfdmod.core.pipeline_yaml import (
    DigestStrategy,
    OpSpec,
    PipelineTemplate,
    _resolve_key,
)
from cfdmod.core.protocols import Storage

# Bump when op semantics or the on-disk layout change in a way that
# invalidates previously generated outputs, independent of the package
# version. Folded into every signature.
FORMAT_VERSION = "1"

DEFAULT_DIGEST: DigestStrategy = "size_mtime"

# Step-level keys that are structural (not params). Everything else on a
# step is a parameter that affects the output and must enter the signature.
_STRUCTURAL_KEYS = frozenset({"id", "kind", "source", "rhs", "recipe", "sources"})

Status = Literal["fresh", "stale", "missing"]


def code_version() -> str:
    """Return the code-version tag folded into signatures.

    Combines the installed package version (best-effort) with
    :data:`FORMAT_VERSION`. A version bump or a format bump changes every
    signature, so outputs regenerate rather than being trusted across an
    upgrade that could have changed their meaning.
    """
    try:
        from importlib.metadata import version

        pkg = version("aerosim-cfdmod")
    except Exception:
        pkg = "unknown"
    return f"{pkg}+fmt{FORMAT_VERSION}"


class OutputStatus(BaseModel):
    """Freshness verdict for one declared output.

    Attributes:
        name: The ``outputs:`` key.
        status: ``fresh`` (up to date), ``stale`` (inputs/params changed),
            or ``missing`` (never stamped).
        reason: Human-readable explanation.
        expected: The freshly computed signature (``None`` if it could not
            be computed, e.g. a required input is absent).
        stored: The signature currently stamped on the output, if any.
    """

    model_config = {"frozen": True}

    name: str
    status: Status
    reason: str
    expected: str | None = None
    stored: str | None = None

    @property
    def is_fresh(self) -> bool:
        return self.status == "fresh"


def _step_id(index: int, step: OpSpec) -> str:
    """Resolve a step's id the same way the runner does."""
    return step.id or f"step_{index}"


def _step_deps(step: OpSpec) -> list[str]:
    """Binding names a step consumes: source (+ rhs) (+ recipe sources)."""
    dump = step.model_dump()
    deps = [dump["source"]]
    rhs = dump.get("rhs")
    if rhs is not None:
        deps.append(rhs)
    sources = dump.get("sources")
    if isinstance(sources, dict):
        deps.extend(str(v) for v in sources.values())
    return deps


def immediate_deps(template: PipelineTemplate) -> dict[str, list[str]]:
    """Map each step id to the binding names it directly consumes."""
    return {_step_id(i, step): _step_deps(step) for i, step in enumerate(template.pipeline)}


def _closure(template: PipelineTemplate, targets: list[str]) -> tuple[set[str], set[str]]:
    """Transitive dependency closure of ``targets`` over the step graph.

    Returns ``(step_ids, input_names)`` reachable from the targets. Unknown
    names (which :func:`validate_template` would already have rejected) are
    ignored here.
    """
    deps = immediate_deps(template)
    input_names = set(template.inputs)
    needed_steps: set[str] = set()
    needed_inputs: set[str] = set()
    stack = list(targets)
    while stack:
        name = stack.pop()
        if name in input_names:
            needed_inputs.add(name)
            continue
        if name in needed_steps:
            continue
        if name in deps:
            needed_steps.add(name)
            stack.extend(deps[name])
    return needed_steps, needed_inputs


def output_closure(template: PipelineTemplate, output_name: str) -> tuple[set[str], set[str]]:
    """Steps + inputs a single declared output depends on."""
    out = template.outputs[output_name]
    return _closure(template, [out.source])


def closure_for_outputs(
    template: PipelineTemplate, output_names: set[str] | list[str]
) -> tuple[set[str], set[str]]:
    """Union of the dependency closures of several outputs."""
    targets = [template.outputs[n].source for n in output_names]
    return _closure(template, targets)


def resolve_strategy(
    template: PipelineTemplate, override: DigestStrategy | None = None
) -> DigestStrategy:
    """Pick the digest strategy: explicit override wins, else the template's."""
    if override is not None:
        return override
    return template.freshness.digest


def _input_strategy(
    template: PipelineTemplate, input_name: str, strategy: DigestStrategy
) -> DigestStrategy:
    return template.freshness.per_input.get(input_name, strategy)


def signature(
    template: PipelineTemplate,
    output_name: str,
    storage: Storage,
    strategy: DigestStrategy = DEFAULT_DIGEST,
) -> str:
    """Compute the freshness signature of one declared output.

    Deterministic across runs and machines: dict keys are sorted, paths are
    normalized to storage keys, and no wall-clock / RNG enters the fold.

    Raises :class:`StorageKeyError` if a required input has no digestible
    object yet (i.e. it has not been produced) -- the caller treats that as
    "cannot be fresh".
    """
    needed_steps, needed_inputs = output_closure(template, output_name)

    steps_repr = []
    for i, step in enumerate(template.pipeline):
        sid = _step_id(i, step)
        if sid not in needed_steps:
            continue
        dump = step.model_dump()
        params = {k: v for k, v in dump.items() if k not in _STRUCTURAL_KEYS}
        steps_repr.append(
            {
                "id": sid,
                "kind": dump.get("kind"),
                "recipe": dump.get("recipe"),
                "source": dump.get("source"),
                "rhs": dump.get("rhs"),
                "sources": dump.get("sources"),
                "params": params,
            }
        )

    inputs_repr = {}
    for name in sorted(needed_inputs):
        spec = template.inputs[name]
        key = _resolve_key(template.root, spec.path)
        strat = _input_strategy(template, name, strategy)
        inputs_repr[name] = {
            "kind": spec.kind,
            "field": spec.field,
            "key": key,
            "digest": storage.digest(key, strat),
        }

    out = template.outputs[output_name]
    payload = {
        "code_version": code_version(),
        "output": {
            "name": output_name,
            "key": _resolve_key(template.root, out.path),
            "format": out.format,
        },
        "steps": steps_repr,
        "inputs": inputs_repr,
    }
    blob = json.dumps(payload, sort_keys=True, separators=(",", ":"), default=str)
    return hashlib.blake2b(blob.encode("utf-8"), digest_size=32).hexdigest()


def output_status(
    template: PipelineTemplate,
    storage: Storage,
    strategy: DigestStrategy | None = None,
) -> dict[str, OutputStatus]:
    """Report ``fresh | stale | missing`` for every declared output.

    Pure inspection -- reads input digests and the stored signatures, runs
    no ops and writes nothing.
    """
    strat = resolve_strategy(template, strategy)
    result: dict[str, OutputStatus] = {}
    for name, out in template.outputs.items():
        out_key = _resolve_key(template.root, out.path)
        try:
            expected = signature(template, name, storage, strat)
        except StorageKeyError as exc:
            result[name] = OutputStatus(
                name=name,
                status="stale",
                reason=f"input unavailable: {exc}",
                expected=None,
                stored=None,
            )
            continue
        stored = storage.read_signature(out_key)
        if stored is None:
            status: Status = "missing"
            reason = "no stored signature (not generated with freshness tracking)"
        elif stored == expected:
            status = "fresh"
            reason = "up to date"
        else:
            status = "stale"
            reason = "inputs or params changed since last run"
        result[name] = OutputStatus(
            name=name, status=status, reason=reason, expected=expected, stored=stored
        )
    return result
