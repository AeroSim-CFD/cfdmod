"""Unit tests for the three time ops: window, translate, rescale."""

from __future__ import annotations

import numpy as np
import pytest

from cfdmod.adapters.memory import MemoryFieldStore
from cfdmod.core import (
    ElementMeta,
    SurfaceDataSource,
    TimeAxis,
    Topology,
)
from cfdmod.core.ops.time import (
    RescaleTimeParams,
    TranslateParams,
    WindowSelectionParams,
    rescale,
    translate,
    window_selection,
)


def _surface(n_timesteps: int = 10) -> SurfaceDataSource:
    verts = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0]], dtype=np.float64)
    tris = np.array([[0, 1, 2]], dtype=np.int32)
    pressure = np.arange(n_timesteps, dtype=np.float64).reshape(1, n_timesteps)
    return SurfaceDataSource(
        time=TimeAxis(initial_time=0.0, timestep_size=0.1, n_timesteps=n_timesteps),
        topology=Topology.triangles(tris, verts),
        elements=ElementMeta(),
        fields=MemoryFieldStore({"pressure": pressure}),
    )


def test_window_selection_slices_time_axis_and_fields():
    ds = _surface(n_timesteps=10)
    out = window_selection(ds, WindowSelectionParams(start=0.3, end=0.7))
    assert out.time.n_timesteps == 5
    assert out.time.initial_time == pytest.approx(0.3)
    np.testing.assert_array_equal(out.fields.read("pressure")[0], np.arange(3, 8))


def test_window_selection_keeps_field_meta():
    ds = _surface()
    out = window_selection(ds, WindowSelectionParams(start=0.0, end=0.2))
    assert out.field_names == ["pressure"]


def test_translate_only_changes_initial_time():
    ds = _surface()
    out = translate(ds, TranslateParams(new_initial_time=5.0))
    assert out.time.initial_time == 5.0
    assert out.time.timestep_size == ds.time.timestep_size
    assert out.time.n_timesteps == ds.time.n_timesteps
    # field arrays unchanged (shared by reference)
    np.testing.assert_array_equal(out.fields.read("pressure"), ds.fields.read("pressure"))


def test_rescale_multiplies_axis_metadata():
    ds = _surface()
    out = rescale(ds, RescaleTimeParams(factor=2.0))
    assert out.time.timestep_size == pytest.approx(0.2)
    assert out.time.initial_time == pytest.approx(0.0)


def test_rescale_rejects_non_positive_factor():
    ds = _surface()
    with pytest.raises(ValueError):
        rescale(ds, RescaleTimeParams(factor=0.0))


def test_window_then_translate_compose():
    ds = _surface(n_timesteps=10)
    win = window_selection(ds, WindowSelectionParams(start=0.4, end=0.6))
    out = translate(win, TranslateParams(new_initial_time=0.0))
    assert out.time.initial_time == 0.0
    assert out.time.n_timesteps == 3
    np.testing.assert_array_equal(out.fields.read("pressure")[0], np.arange(4, 7))
