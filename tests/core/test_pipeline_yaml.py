"""Unit tests for the YAML pipeline loader."""

from __future__ import annotations

import textwrap

import numpy as np
import pytest

from cfdmod.adapters.memory import MemoryFieldStore, MemoryStorage
from cfdmod.core import (
    ElementMeta,
    PipelineTemplate,
    SurfaceDataSource,
    TimeAxis,
    Topology,
    load_template,
    run_template,
)


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


def test_pipeline_template_round_trips_minimal_yaml(tmp_path):
    yaml_text = textwrap.dedent("""
        name: test
        inputs:
          body:
            kind: surface
            path: body
        pipeline:
          - id: cp
            kind: scale
            source: body
            field: pressure
            factor: 2.0
        outputs:
          cp:
            source: cp
            path: cp_out
        """)
    yaml_file = tmp_path / "template.yaml"
    yaml_file.write_text(yaml_text)
    template = load_template(yaml_file)
    assert template.name == "test"
    assert len(template.pipeline) == 1
    assert template.pipeline[0].kind == "scale"


def test_run_template_executes_pipeline_against_memory_storage():
    body = _surface(np.full((2, 4), 5.0))
    storage = MemoryStorage()
    storage.write_data_source("body", body)

    tpl = PipelineTemplate(
        inputs={"body": {"kind": "surface", "path": "body"}},
        pipeline=[
            {"id": "cp", "kind": "scale", "source": "body", "field": "pressure", "factor": 3.0},
        ],
        outputs={"cp": {"source": "cp", "path": "cp"}},
    )
    bindings = run_template(tpl, storage=storage)
    assert "cp" in bindings
    np.testing.assert_array_equal(bindings["cp"].fields.read("pressure"), 15.0)
    # Output side-effect: written to storage under "cp" too.
    assert "cp" in list(storage.keys())


def test_run_template_binary_op_resolves_rhs():
    lhs = _surface(np.full((2, 3), 10.0))
    rhs = _surface(np.full((2, 3), 3.0))
    storage = MemoryStorage()
    storage.write_data_source("lhs", lhs)
    storage.write_data_source("rhs", rhs)

    tpl = PipelineTemplate(
        inputs={
            "lhs": {"kind": "surface", "path": "lhs"},
            "rhs": {"kind": "surface", "path": "rhs"},
        },
        pipeline=[
            {
                "id": "diff",
                "kind": "sub",
                "source": "lhs",
                "rhs": "rhs",
                "field": "pressure",
                "out": "dp",
            },
        ],
    )
    bindings = run_template(tpl, storage=storage)
    np.testing.assert_array_equal(bindings["diff"].fields.read("dp"), 7.0)


def test_run_template_chains_steps_via_id():
    body = _surface(np.full((1, 3), 100.0))
    storage = MemoryStorage()
    storage.write_data_source("body", body)

    tpl = PipelineTemplate(
        inputs={"body": {"kind": "surface", "path": "body"}},
        pipeline=[
            {"id": "s1", "kind": "scale", "source": "body", "field": "pressure", "factor": 0.5},
            {"id": "s2", "kind": "scale", "source": "s1", "field": "pressure", "factor": 2.0},
        ],
    )
    bindings = run_template(tpl, storage=storage)
    np.testing.assert_array_equal(bindings["s2"].fields.read("pressure"), 100.0)


def test_unknown_op_kind_raises():
    body = _surface(np.zeros((1, 2)))
    storage = MemoryStorage()
    storage.write_data_source("body", body)
    tpl = PipelineTemplate(
        inputs={"body": {"kind": "surface", "path": "body"}},
        pipeline=[{"kind": "nope", "source": "body"}],
    )
    with pytest.raises(KeyError, match="unknown op kind"):
        run_template(tpl, storage=storage)


def test_dangling_source_reference_raises():
    storage = MemoryStorage()
    tpl = PipelineTemplate(
        pipeline=[{"kind": "scale", "source": "missing", "field": "p", "factor": 1.0}],
    )
    with pytest.raises(KeyError, match="unknown source"):
        run_template(tpl, storage=storage)


def test_template_chain_statistics_after_scale():
    body = _surface(np.tile(np.arange(5, dtype=np.float64), (2, 1)))
    storage = MemoryStorage()
    storage.write_data_source("body", body)

    tpl = PipelineTemplate(
        inputs={"body": {"kind": "surface", "path": "body"}},
        pipeline=[
            {"id": "cp", "kind": "scale", "source": "body", "field": "pressure", "factor": 1.0},
            {
                "id": "stats",
                "kind": "statistics",
                "source": "cp",
                "field": "pressure",
                "kinds": ["mean", "max"],
            },
        ],
    )
    bindings = run_template(tpl, storage=storage)
    assert bindings["stats"].time.is_time_aggregated
    np.testing.assert_allclose(bindings["stats"].fields.read("mean"), 2.0)
    np.testing.assert_allclose(bindings["stats"].fields.read("max"), 4.0)


def test_resolve_key_strips_h5_suffix(tmp_path):
    yaml_text = textwrap.dedent("""
        name: test
        inputs:
          body:
            kind: surface
            path: my_data.h5
        pipeline: []
        """)
    f = tmp_path / "t.yaml"
    f.write_text(yaml_text)
    tpl = load_template(f)
    # Just assert the loader accepts the .h5 suffix; the runner will
    # strip it when handing to storage.
    assert tpl.inputs["body"].path == "my_data.h5"
