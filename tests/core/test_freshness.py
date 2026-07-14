"""Tests for output-staleness detection (cfdmod.core.freshness)."""

from __future__ import annotations

import os

import numpy as np
import pytest

from cfdmod.adapters.memory import MemoryFieldStore, MemoryStorage
from cfdmod.adapters.xdmf_h5 import XdmfH5Storage
from cfdmod.core import (
    ElementMeta,
    PipelineTemplate,
    SurfaceDataSource,
    TimeAxis,
    Topology,
    output_status,
    run_template,
)
from cfdmod.core import freshness as fr


def _surface(values: np.ndarray, dt: float = 0.1) -> SurfaceDataSource:
    n_elements, n_t = values.shape
    verts = np.tile(np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0]]), (n_elements, 1)).astype(
        np.float64
    )
    tris = np.arange(n_elements * 3).reshape(n_elements, 3).astype(np.int32)
    return SurfaceDataSource(
        time=TimeAxis(initial_time=0.0, timestep_size=dt, n_timesteps=n_t),
        topology=Topology.triangles(tris, verts),
        elements=ElementMeta(),
        fields=MemoryFieldStore({"pressure": values.astype(np.float64)}),
    )


def _scale_template(factor: float = 3.0) -> PipelineTemplate:
    return PipelineTemplate(
        inputs={"body": {"kind": "surface", "path": "body"}},
        pipeline=[
            {
                "id": "cp",
                "kind": "scale",
                "source": "body",
                "field": "pressure",
                "factor": factor,
            }
        ],
        outputs={"cp": {"source": "cp", "path": "cp"}},
    )


# --- dependency closure ----------------------------------------------------


def test_output_closure_is_transitive_and_prunes_unrelated_steps():
    tpl = PipelineTemplate(
        inputs={
            "a": {"kind": "surface", "path": "a"},
            "b": {"kind": "surface", "path": "b"},
        },
        pipeline=[
            {"id": "s1", "kind": "scale", "source": "a", "field": "pressure", "factor": 2.0},
            {"id": "s2", "kind": "scale", "source": "s1", "field": "pressure", "factor": 2.0},
            {"id": "u1", "kind": "scale", "source": "b", "field": "pressure", "factor": 9.0},
        ],
        outputs={
            "out_s2": {"source": "s2", "path": "o1"},
            "out_u1": {"source": "u1", "path": "o2"},
        },
    )
    steps, inputs = fr.output_closure(tpl, "out_s2")
    assert steps == {"s1", "s2"}
    assert inputs == {"a"}

    all_steps, all_inputs = fr.closure_for_outputs(tpl, {"out_s2", "out_u1"})
    assert all_steps == {"s1", "s2", "u1"}
    assert all_inputs == {"a", "b"}


def test_closure_includes_binary_rhs():
    tpl = PipelineTemplate(
        inputs={
            "lhs": {"kind": "surface", "path": "lhs"},
            "rhs": {"kind": "surface", "path": "rhs"},
        },
        pipeline=[
            {
                "id": "d",
                "kind": "sub",
                "source": "lhs",
                "rhs": "rhs",
                "field": "pressure",
                "out": "dp",
            }
        ],
        outputs={"d": {"source": "d", "path": "d"}},
    )
    steps, inputs = fr.output_closure(tpl, "d")
    assert steps == {"d"}
    assert inputs == {"lhs", "rhs"}


# --- signature -------------------------------------------------------------


def test_signature_is_stable_across_runs():
    storage = MemoryStorage()
    storage.write_data_source("body", _surface(np.full((2, 3), 5.0)))
    tpl = _scale_template()
    s1 = fr.signature(tpl, "cp", storage, "content")
    s2 = fr.signature(tpl, "cp", storage, "content")
    assert s1 == s2


def test_signature_changes_when_param_changes():
    storage = MemoryStorage()
    storage.write_data_source("body", _surface(np.full((2, 3), 5.0)))
    s_a = fr.signature(_scale_template(3.0), "cp", storage, "content")
    s_b = fr.signature(_scale_template(4.0), "cp", storage, "content")
    assert s_a != s_b


def test_signature_changes_when_input_changes():
    storage = MemoryStorage()
    tpl = _scale_template()
    storage.write_data_source("body", _surface(np.full((2, 3), 5.0)))
    s_before = fr.signature(tpl, "cp", storage, "content")
    storage.write_data_source("body", _surface(np.full((2, 3), 6.0)))
    s_after = fr.signature(tpl, "cp", storage, "content")
    assert s_before != s_after


def test_signature_changes_when_format_version_bumps(monkeypatch):
    storage = MemoryStorage()
    storage.write_data_source("body", _surface(np.full((2, 3), 5.0)))
    tpl = _scale_template()
    s_v1 = fr.signature(tpl, "cp", storage, "content")
    monkeypatch.setattr(fr, "FORMAT_VERSION", "999")
    s_v2 = fr.signature(tpl, "cp", storage, "content")
    assert s_v1 != s_v2


def test_signature_missing_input_raises():
    from cfdmod.core.errors import StorageKeyError

    storage = MemoryStorage()  # body never written
    with pytest.raises(StorageKeyError):
        fr.signature(_scale_template(), "cp", storage, "content")


# --- output_status ---------------------------------------------------------


def test_output_status_missing_then_fresh_then_stale_memory():
    storage = MemoryStorage()
    storage.write_data_source("body", _surface(np.full((2, 3), 5.0)))
    tpl = _scale_template()

    # Never run -> missing.
    st = output_status(tpl, storage)
    assert st["cp"].status == "missing"

    # Run stamps a signature -> fresh.
    run_template(tpl, storage=storage)
    st = output_status(tpl, storage)
    assert st["cp"].status == "fresh"

    # Change an input -> stale.
    storage.write_data_source("body", _surface(np.full((2, 3), 9.0)))
    st = output_status(tpl, storage)
    assert st["cp"].status == "stale"


# --- skip_fresh run mode ---------------------------------------------------


class _SpyStorage(MemoryStorage):
    """MemoryStorage that records which keys were (re)written."""

    def __init__(self):
        super().__init__()
        self.writes = []

    def write_data_source(self, key, ds):
        self.writes.append(key)
        super().write_data_source(key, ds)


def test_skip_fresh_no_op_when_all_fresh():
    storage = _SpyStorage()
    storage.write_data_source("body", _surface(np.full((2, 3), 5.0)))
    tpl = _scale_template()

    run_template(tpl, storage=storage)
    storage.writes.clear()

    bindings = run_template(tpl, storage=storage, skip_fresh=True)
    assert bindings == {}
    assert storage.writes == []  # nothing rewritten


def test_skip_fresh_recomputes_only_stale_output():
    storage = _SpyStorage()
    storage.write_data_source("a", _surface(np.full((2, 3), 1.0)))
    storage.write_data_source("b", _surface(np.full((2, 3), 1.0)))
    tpl = PipelineTemplate(
        inputs={
            "a": {"kind": "surface", "path": "a"},
            "b": {"kind": "surface", "path": "b"},
        },
        pipeline=[
            {"id": "sa", "kind": "scale", "source": "a", "field": "pressure", "factor": 2.0},
            {"id": "sb", "kind": "scale", "source": "b", "field": "pressure", "factor": 2.0},
        ],
        outputs={
            "oa": {"source": "sa", "path": "oa"},
            "ob": {"source": "sb", "path": "ob"},
        },
    )
    run_template(tpl, storage=storage)

    # Touch only input b -> only ob should be recomputed.
    storage.write_data_source("b", _surface(np.full((2, 3), 7.0)))
    storage.writes.clear()
    run_template(tpl, storage=storage, skip_fresh=True)
    assert storage.writes == ["ob"]


def test_skip_fresh_requires_supporting_backend():
    class _Bare:
        def read_data_source(self, key):
            raise KeyError(key)

        def write_data_source(self, key, ds):
            pass

        def keys(self):
            return []

    from cfdmod.core.errors import TemplateError

    with pytest.raises(TemplateError, match="skip_fresh"):
        run_template(_scale_template(), storage=_Bare(), skip_fresh=True)


# --- XdmfH5 digest + stamping ----------------------------------------------


def test_xdmf_digest_size_mtime_flips_on_touch(tmp_path):
    storage = XdmfH5Storage(tmp_path)
    storage.write_data_source("body", _surface(np.full((2, 3), 5.0)))
    d1 = storage.digest("body", "size_mtime")
    # Bump mtime by a second so st_mtime_ns changes deterministically.
    st = (tmp_path / "body.h5").stat()
    os.utime(tmp_path / "body.h5", ns=(st.st_atime_ns, st.st_mtime_ns + 1_000_000_000))
    d2 = storage.digest("body", "size_mtime")
    assert d1 != d2


def test_xdmf_digest_content_stable_across_identical_copy(tmp_path):
    s1 = XdmfH5Storage(tmp_path / "a")
    s2 = XdmfH5Storage(tmp_path / "b")
    ds = _surface(np.full((2, 3), 5.0))
    s1.write_data_source("body", ds)
    s2.write_data_source("body", ds)
    assert s1.digest("body", "content") == s2.digest("body", "content")


def test_xdmf_digest_missing_key_raises(tmp_path):
    from cfdmod.core.errors import StorageKeyError

    with pytest.raises(StorageKeyError):
        XdmfH5Storage(tmp_path).digest("nope", "size_mtime")


def test_xdmf_run_stamps_and_status_roundtrips(tmp_path):
    storage = XdmfH5Storage(tmp_path)
    storage.write_data_source("body", _surface(np.full((2, 3), 5.0)))
    tpl = PipelineTemplate(
        root=str(tmp_path),
        inputs={"body": {"kind": "surface", "path": "body"}},
        pipeline=[
            {"id": "cp", "kind": "scale", "source": "body", "field": "pressure", "factor": 2.0}
        ],
        outputs={"cp": {"source": "cp", "path": "cp"}},
    )
    assert output_status(tpl, storage)["cp"].status == "missing"
    run_template(tpl, storage=storage)
    assert output_status(tpl, storage)["cp"].status == "fresh"

    # Stamping must not corrupt the readable data source.
    back = storage.read_data_source("cp")
    np.testing.assert_allclose(back.fields.read("pressure"), 10.0)


def test_xdmf_read_signature_none_when_unstamped(tmp_path):
    storage = XdmfH5Storage(tmp_path)
    storage.write_data_source("body", _surface(np.full((2, 3), 5.0)))
    assert storage.read_signature("body") is None
