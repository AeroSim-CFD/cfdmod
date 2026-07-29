"""What a run keeps, what it writes, and what it lets go.

`run_template` used to return every binding it ever computed - inputs and all
intermediates - and each kept its field arrays reachable. In a notebook that is
a feature; in a service job it means the peak of a run is the sum of everything
it computed rather than its widest live set.
"""

from __future__ import annotations

import gc
import tracemalloc
import weakref

import numpy as np
import pytest

from cfdmod.adapters.memory import MemoryFieldStore, MemoryStorage
from cfdmod.core import ElementMeta, SurfaceDataSource, TimeAxis, Topology
from cfdmod.core.errors import TemplateError
from cfdmod.core.pipeline_yaml import PipelineTemplate, run_template, validate_template

pytestmark = pytest.mark.unit


def _surface(n_elements: int = 6, n_timesteps: int = 8) -> SurfaceDataSource:
    rng = np.random.default_rng(0)
    verts = rng.random((n_elements * 3, 3))
    tris = np.arange(n_elements * 3, dtype=np.int32).reshape(n_elements, 3)
    return SurfaceDataSource(
        time=TimeAxis(initial_time=0.0, timestep_size=0.1, n_timesteps=n_timesteps),
        topology=Topology.triangles(tris, verts),
        elements=ElementMeta(),
        fields=MemoryFieldStore(
            {"pressure": rng.random((n_elements, n_timesteps)).astype(np.float32)}
        ),
    )


def _template(outputs: dict) -> PipelineTemplate:
    return PipelineTemplate.model_validate(
        {
            "name": "outputs",
            "inputs": {"body": {"kind": "surface", "path": "body", "field": "pressure"}},
            "pipeline": [
                {
                    "id": "a",
                    "kind": "scale",
                    "source": "body",
                    "field": "pressure",
                    "factor": 2.0,
                    "out": "cp",
                },
                {
                    "id": "b",
                    "kind": "scale",
                    "source": "a",
                    "field": "cp",
                    "factor": 0.5,
                    "out": "cp_half",
                },
            ],
            "outputs": outputs,
        }
    )


def _storage(n_elements: int = 6, n_timesteps: int = 8) -> MemoryStorage:
    storage = MemoryStorage()
    storage.write_data_source("body", _surface(n_elements, n_timesteps))
    return storage


def test_intermediates_are_not_returned_by_default():
    """`a` is read only by `b`; nothing downstream can ask for it again."""
    result = run_template(_template({"o": {"source": "b", "path": "out"}}), storage=_storage())

    assert "b" in result
    assert "a" not in result


def test_return_all_keeps_them():
    result = run_template(
        _template({"o": {"source": "b", "path": "out"}}), storage=_storage(), return_all=True
    )
    assert {"a", "b"} <= set(result)


def test_a_template_with_no_outputs_keeps_everything():
    """Nothing to select on -- that template is being run for its intermediates."""
    result = run_template(_template({}), storage=_storage())
    assert {"a", "b"} <= set(result)


def test_a_released_intermediate_is_actually_unreachable():
    """A weakref, not an absent dict key.

    Dropping the name while something else still holds the object would look
    identical from the outside and free nothing.
    """
    storage = _storage()
    full = run_template(
        _template({"o": {"source": "b", "path": "out"}}), storage=storage, return_all=True
    )
    ref = weakref.ref(full["a"])
    del full
    gc.collect()
    assert ref() is None


def test_persist_false_computes_without_writing():
    storage = _storage()
    before = set(storage.keys())
    result = run_template(
        _template({"o": {"source": "b", "path": "out", "persist": False}}), storage=storage
    )

    assert set(storage.keys()) == before
    assert "b" in result  # it was still computed and handed back


def test_hold_false_writes_and_drops():
    storage = _storage()
    result = run_template(
        _template({"o": {"source": "b", "path": "out", "hold": False}}), storage=storage
    )

    assert "out" in set(storage.keys())
    assert "b" not in result


def test_hold_false_is_ignored_when_another_output_holds_the_same_source():
    """Two outputs, one source: dropping it would break the one that holds."""
    storage = _storage()
    result = run_template(
        _template(
            {
                "written": {"source": "b", "path": "out", "hold": False},
                "kept": {"source": "b", "path": "out2"},
            }
        ),
        storage=storage,
    )
    assert "b" in result


def test_persist_and_hold_both_false_is_rejected():
    """Computing something and throwing it away is a template bug, not a mode."""
    template = _template({"o": {"source": "b", "path": "out", "persist": False, "hold": False}})
    with pytest.raises(TemplateError, match="computed and discarded"):
        validate_template(template)


def test_defaults_are_unchanged_behaviour():
    storage = _storage()
    result = run_template(_template({"o": {"source": "b", "path": "out"}}), storage=storage)
    assert "out" in set(storage.keys())
    assert "b" in result


def test_releasing_intermediates_lowers_the_peak():
    """Measured. A four-step chain over a large input holds one stage, not four.

    The steps rewrite ``pressure`` in place rather than using ``out:``. That is
    not incidental: an op given ``out:`` *adds* a field, so every later stage
    keeps the earlier ones reachable through its own store and dropping the
    binding frees nothing. Releasing intermediates only recovers memory for
    ops that replace rather than accumulate -- which is the common shape
    (Cp -> per-floor Cf/Cm) but is worth knowing.
    """
    n_elements, n_timesteps = 20_000, 400
    template = PipelineTemplate.model_validate(
        {
            "name": "chain",
            "inputs": {"body": {"kind": "surface", "path": "body", "field": "pressure"}},
            "pipeline": [
                {
                    "id": f"s{i}",
                    "kind": "scale",
                    "source": "body" if i == 0 else f"s{i - 1}",
                    "field": "pressure",
                    "factor": 1.5,
                }
                for i in range(4)
            ],
            "outputs": {"o": {"source": "s3", "path": "out", "persist": False}},
        }
    )

    def peak(**kwargs) -> int:
        storage = _storage(n_elements, n_timesteps)
        tracemalloc.start()
        try:
            run_template(template, storage=storage, **kwargs)
            return tracemalloc.get_traced_memory()[1]
        finally:
            tracemalloc.stop()

    all_kept = peak(return_all=True)
    released = peak()

    field_bytes = n_elements * n_timesteps * 4
    assert released < all_kept, (
        f"releasing intermediates peaked at {released / 1e6:.1f} MB, "
        f"not below {all_kept / 1e6:.1f} MB with everything kept"
    )
    # Four stages kept is roughly four copies; releasing should stay near one.
    assert released < all_kept - field_bytes
