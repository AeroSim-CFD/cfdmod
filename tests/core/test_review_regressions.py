"""Regression tests for fixes from the v3 quality review.

Each test pins a behavior that was previously silently wrong: a
functional update bypassing validators, a window silently clamping, a
zero divisor producing inf, a typo'd op field being dropped, and the
template loader not validating up front.
"""

from __future__ import annotations

import numpy as np
import pytest

from cfdmod.adapters.memory import MemoryFieldStore, MemoryStorage
from cfdmod.core import (
    ElementMeta,
    PointsDataSource,
    SurfaceDataSource,
    TimeAxis,
    Topology,
    algebra,
)
from cfdmod.core.ops.time import WindowSelectionParams, window_selection
from cfdmod.core.pipeline_yaml import PipelineTemplate, run_template, validate_template


def _surface(n_t: int = 3) -> SurfaceDataSource:
    verts = np.array([[0, 0, 0], [1, 0, 0], [1, 1, 0], [0, 1, 0]], dtype=np.float64)
    tris = np.array([[0, 1, 2], [0, 2, 3]], dtype=np.int32)
    return SurfaceDataSource(
        time=TimeAxis(initial_time=0.0, timestep_size=1.0, n_timesteps=n_t),
        topology=Topology.triangles(tris, verts),
        elements=ElementMeta(),
        fields=MemoryFieldStore({"cp": np.zeros((2, n_t))}),
    )


# --- Functional-update validation (model_copy no longer skips validators) ---


def test_with_time_rejects_inconsistent_axis():
    ds = _surface(n_t=3)
    with pytest.raises(ValueError):
        ds.with_time(TimeAxis(initial_time=0.0, timestep_size=1.0, n_timesteps=99))


def test_with_field_rejects_wrong_n_elements():
    ds = _surface(n_t=3)
    with pytest.raises(ValueError):
        ds.with_field("bad", np.zeros((99, 3)))


def test_with_field_accepts_consistent_field():
    ds = _surface(n_t=3)
    out = ds.with_field("cp2", np.ones((2, 3)))
    assert np.allclose(out.fields.read("cp2"), 1.0)


# --- TimeAxis.window overlap check ---


def test_window_out_of_range_raises():
    ds = _surface(n_t=3)  # times [0, 1, 2]
    with pytest.raises(ValueError):
        window_selection(ds, WindowSelectionParams(start=50.0, end=60.0))


def test_window_in_range_ok():
    ds = _surface(n_t=3)
    out = window_selection(ds, WindowSelectionParams(start=1.0, end=2.0))
    assert out.time.n_timesteps == 2


# --- Division by scalar zero ---


def test_div_by_scalar_zero_raises():
    ds = _surface(n_t=3)
    with pytest.raises(ValueError):
        algebra.div(ds, 0, field="cp")


# --- OpParams extra="forbid" ---


def test_typoed_op_field_rejected():
    from cfdmod.core.ops.field.algebra import ScaleParams

    with pytest.raises(Exception):
        ScaleParams.model_validate({"field": "cp", "factor": 2.0, "factr": 9})


# --- validate_template static DAG checks ---


def test_validate_template_dangling_source():
    t = PipelineTemplate(
        pipeline=[{"id": "a", "kind": "scale", "source": "nope", "field": "cp", "factor": 2.0}]
    )
    with pytest.raises(KeyError):
        validate_template(t)


def test_validate_template_duplicate_id():
    t = PipelineTemplate(
        inputs={"body": {"kind": "surface", "path": "b"}},
        pipeline=[
            {"id": "x", "kind": "scale", "source": "body", "field": "cp", "factor": 2.0},
            {"id": "x", "kind": "scale", "source": "x", "field": "cp", "factor": 2.0},
        ],
    )
    with pytest.raises(ValueError):
        validate_template(t)


def test_validate_template_rhs_on_unary():
    t = PipelineTemplate(
        inputs={"body": {"kind": "surface", "path": "b"}},
        pipeline=[
            {
                "id": "x",
                "kind": "scale",
                "source": "body",
                "rhs": "body",
                "field": "cp",
                "factor": 2.0,
            }
        ],
    )
    with pytest.raises(ValueError):
        validate_template(t)


# --- run_template honors declared input kind ---


def test_run_template_input_kind_mismatch_raises():
    # Store a points source but declare the input as a surface.
    pts = PointsDataSource(
        time=TimeAxis(initial_time=0.0, timestep_size=1.0, n_timesteps=2),
        topology=Topology.points(np.array([[0.0, 0.0, 0.0]])),
        elements=ElementMeta(),
        fields=MemoryFieldStore({"cp": np.zeros((1, 2))}),
    )
    storage = MemoryStorage()
    storage.write_data_source("thing", pts)
    t = PipelineTemplate(inputs={"body": {"kind": "surface", "path": "thing"}})
    with pytest.raises(ValueError):
        run_template(t, storage=storage)
