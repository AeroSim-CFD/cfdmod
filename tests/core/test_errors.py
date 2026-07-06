"""Typed exception hierarchy (issue #147).

The typed errors must be catchable by their new types AND by the builtin
they replaced, so a consumer can migrate gradually and existing
``except (KeyError, ValueError)`` handlers keep working.
"""

from __future__ import annotations

import pytest

from cfdmod.core import (
    CfdmodError,
    OpError,
    PipelineTemplate,
    StorageKeyError,
    TemplateError,
    TemplateReferenceError,
)
from cfdmod.core.pipeline_yaml import validate_template

pytestmark = pytest.mark.unit


def test_backward_compatible_bases():
    assert issubclass(TemplateError, (CfdmodError, ValueError))
    assert issubclass(TemplateReferenceError, (TemplateError, KeyError))
    assert issubclass(OpError, (CfdmodError, RuntimeError))
    assert issubclass(StorageKeyError, (CfdmodError, KeyError))


def test_messages_are_plain_not_repr_wrapped():
    """KeyError-derived errors must not wrap their message in quotes."""
    err = TemplateReferenceError("no such source 'x'")
    assert str(err) == "no such source 'x'"
    assert str(StorageKeyError("missing key foo")) == "missing key foo"


def test_unknown_op_raises_reference_error_catchable_as_keyerror():
    tpl = PipelineTemplate(
        name="t",
        root="/tmp",
        inputs={"body": {"kind": "surface", "path": "body"}},
        pipeline=[{"id": "s", "kind": "no_such_op", "source": "body"}],
    )
    with pytest.raises(TemplateReferenceError):
        validate_template(tpl)
    with pytest.raises(KeyError):  # backward compatibility
        validate_template(tpl)


def test_contract_violation_raises_template_error_catchable_as_valueerror():
    tpl = PipelineTemplate(
        name="t",
        root="/tmp",
        inputs={"body": {"kind": "surface", "path": "body"}},
        pipeline=[
            {"id": "f", "kind": "force_contribution", "source": "body", "nominal_area": 1.0}
        ],
    )
    with pytest.raises(TemplateError):
        validate_template(tpl)
    with pytest.raises(ValueError):  # backward compatibility
        validate_template(tpl)


def test_op_error_carries_step_id_and_kind():
    """An op that raises at run time is wrapped as OpError with context."""
    from typing import Literal

    import numpy as np

    from cfdmod.adapters.memory import MemoryFieldStore, MemoryStorage
    from cfdmod.core import (
        DataSource,
        ElementMeta,
        SurfaceDataSource,
        TimeAxis,
        Topology,
        register_op,
        run_template,
    )
    from cfdmod.core.ops import OpParams

    class BoomParams(OpParams):
        kind: Literal["boom"] = "boom"

    def boom(ds: DataSource, p: BoomParams) -> DataSource:
        raise RuntimeError("kaboom")

    register_op("boom", boom, BoomParams, arity="unary")
    try:
        verts = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0]], dtype=np.float64)
        tris = np.array([[0, 1, 2]], dtype=np.int32)
        ds = SurfaceDataSource(
            time=TimeAxis(initial_time=0.0, timestep_size=0.1, n_timesteps=4),
            topology=Topology.triangles(tris, verts),
            elements=ElementMeta(),
            fields=MemoryFieldStore({"pressure": np.ones((1, 4))}),
        )
        storage = MemoryStorage()
        storage.write_data_source("body", ds)
        tpl = PipelineTemplate(
            name="t",
            root="",
            inputs={"body": {"kind": "surface", "path": "body"}},
            pipeline=[{"id": "b", "kind": "boom", "source": "body"}],
        )
        with pytest.raises(OpError) as ei:
            run_template(tpl, storage=storage)
        assert ei.value.step_id == "b"
        assert ei.value.op_kind == "boom"
    finally:
        from cfdmod.core.pipeline_yaml import OP_REGISTRY

        OP_REGISTRY.pop("boom", None)


def test_storage_key_error():
    from cfdmod.adapters.memory import MemoryStorage

    with pytest.raises(StorageKeyError):
        MemoryStorage().read_data_source("absent")
    with pytest.raises(KeyError):  # backward compatibility
        MemoryStorage().read_data_source("absent")
