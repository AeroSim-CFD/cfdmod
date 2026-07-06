"""Property-based pipeline-loader tests (issue #141 phase 3).

The static check (:func:`validate_template`) and the runtime
(:func:`run_template`, which calls ``validate_template`` first) must not
diverge: a template the static check accepts must run, and one it rejects must
fail the runtime too. We assert that agreement generatively by starting from a
valid template and applying one of several structural mutations, then checking
that both entry points make the *same* accept/reject decision.

The valid case is executed end-to-end against a :class:`MemoryStorage` preloaded
with the declared input, so "accepted by validate" is confirmed to actually run
rather than merely pass the static walk.
"""

from __future__ import annotations

import numpy as np
import pytest
from hypothesis import given
from hypothesis import strategies as st

from cfdmod.adapters.memory import MemoryFieldStore, MemoryStorage
from cfdmod.core.data_source import PointsDataSource
from cfdmod.core.pipeline_yaml import (
    InputSpec,
    OpSpec,
    OutputSpec,
    PipelineTemplate,
    run_template,
    validate_template,
)
from cfdmod.core.time_axis import TimeAxis
from cfdmod.core.topology import ElementMeta, Topology

pytestmark = pytest.mark.property

MUTATIONS = [
    "valid",
    "unknown_kind",
    "dangling_source",
    "rhs_on_unary",
    "duplicate_id",
    "missing_param",
    "dangling_output",
]


def _storage_with_input() -> MemoryStorage:
    storage = MemoryStorage()
    src = PointsDataSource(
        time=TimeAxis(initial_time=0.0, timestep_size=0.0, n_timesteps=0),
        topology=Topology.points(np.zeros((2, 3), dtype=np.float64)),
        elements=ElementMeta(),
        fields=MemoryFieldStore({"v": np.array([1.0, 2.0], dtype=np.float64)}),
    )
    storage.write_data_source("inp", src)
    return storage


def _build(mutation: str, factor: float) -> PipelineTemplate:
    """A valid single-step scale template, optionally broken one way."""
    step = OpSpec(id="scaled", kind="scale", source="inp", field="v", factor=factor)
    pipeline = [step]
    outputs = {"out": OutputSpec(source="scaled", path="out")}

    if mutation == "unknown_kind":
        step.kind = "no_such_op"
    elif mutation == "dangling_source":
        step.source = "ghost"
    elif mutation == "rhs_on_unary":
        step.rhs = "inp"  # scale is unary
    elif mutation == "duplicate_id":
        pipeline = [step, OpSpec(id="scaled", kind="scale", source="inp", field="v", factor=1.0)]
    elif mutation == "missing_param":
        step = OpSpec(id="scaled", kind="scale", source="inp", field="v")  # no factor
        pipeline = [step]
    elif mutation == "dangling_output":
        outputs = {"out": OutputSpec(source="ghost", path="out")}

    return PipelineTemplate(
        name="t",
        root=None,
        inputs={"inp": InputSpec(kind="points", path="inp")},
        pipeline=pipeline,
        outputs=outputs,
    )


def _raises(thunk) -> bool:
    try:
        thunk()
        return False
    except Exception:
        return True


@given(
    mutation=st.sampled_from(MUTATIONS),
    factor=st.floats(min_value=-1e3, max_value=1e3, allow_nan=False, allow_infinity=False),
)
def test_validate_and_run_agree(mutation: str, factor: float) -> None:
    template = _build(mutation, factor)
    storage = _storage_with_input()

    validate_raised = _raises(lambda: validate_template(template))
    run_raised = _raises(lambda: run_template(template, storage=storage))

    # No divergence: the static check and the runtime reach the same verdict.
    assert validate_raised == run_raised
    # And the sole valid shape must be accepted (both must succeed).
    assert (mutation == "valid") == (not validate_raised)
