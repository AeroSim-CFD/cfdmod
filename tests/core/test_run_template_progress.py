"""Progress reporting and cooperative cancellation on a template run."""

from __future__ import annotations

import numpy as np
import pytest

from cfdmod.adapters.memory import MemoryFieldStore, MemoryStorage
from cfdmod.core import ElementMeta, SurfaceDataSource, TimeAxis, Topology
from cfdmod.core.pipeline_yaml import PipelineTemplate, run_template
from cfdmod.core.progress import RunCancelled, RunEvent

pytestmark = pytest.mark.unit


def _surface(n_elements: int = 6, n_timesteps: int = 32) -> SurfaceDataSource:
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


def _template() -> PipelineTemplate:
    return PipelineTemplate.model_validate(
        {
            "name": "progress",
            "inputs": {"body": {"kind": "surface", "path": "body", "field": "pressure"}},
            "pipeline": [
                {
                    "id": "scaled",
                    "kind": "scale",
                    "source": "body",
                    "field": "pressure",
                    "factor": 2.0,
                    "out": "cp",
                },
                {
                    "id": "halved",
                    "kind": "scale",
                    "source": "scaled",
                    "field": "cp",
                    "factor": 0.5,
                    "out": "cp_half",
                },
            ],
            "outputs": {"result": {"source": "halved", "path": "out"}},
        }
    )


def _storage() -> MemoryStorage:
    storage = MemoryStorage()
    storage.write_data_source("body", _surface())
    return storage


def test_every_phase_reports():
    """Loading and writing are reported too.

    A bar built only from the step phase sits at 0% while inputs load and jumps
    to 100% before the write - which on a big case is where a real share of the
    wall clock goes.
    """
    events: list[RunEvent] = []
    run_template(_template(), storage=_storage(), on_progress=events.append)

    assert [e.phase for e in events] == ["load", "step", "step", "write"]
    assert [e.name for e in events] == ["body", "scaled", "halved", "result"]
    assert [e.op_kind for e in events] == [None, "scale", "scale", None]


def test_event_carries_position_and_fraction():
    events: list[RunEvent] = []
    run_template(_template(), storage=_storage(), on_progress=events.append)

    steps = [e for e in events if e.phase == "step"]
    assert [(e.index, e.total) for e in steps] == [(0, 2), (1, 2)]
    assert [e.fraction for e in steps] == [0.5, 1.0]
    assert "[step] 1/2 scaled (scale)" == steps[0].describe()


def test_chunked_run_reports_the_window():
    """Per-step events repeat per window, and say which window they are in."""
    events: list[RunEvent] = []
    run_template(_template(), storage=_storage(), chunk_size=8, on_progress=events.append)

    steps = [e for e in events if e.phase == "step"]
    # 32 timesteps / 8 = 4 windows x 2 steps.
    assert len(steps) == 8
    assert [e.window for e in steps[:2]] == [0, 0]
    assert {e.n_windows for e in steps} == {4}
    assert "window 1/4" in steps[0].describe()


def test_unchunked_events_carry_no_window():
    events: list[RunEvent] = []
    run_template(_template(), storage=_storage(), on_progress=events.append)
    assert all(e.window is None and e.n_windows is None for e in events)


def test_cancel_stops_the_run_and_says_where():
    """Cancelling before the second step must not run it."""
    seen: list[str] = []

    def cancel() -> bool:
        return len(seen) >= 2  # after load + first step

    def on_progress(event: RunEvent) -> None:
        seen.append(event.name)

    with pytest.raises(RunCancelled) as excinfo:
        run_template(_template(), storage=_storage(), on_progress=on_progress, cancel=cancel)

    assert excinfo.value.phase == "step"
    assert excinfo.value.name == "halved"
    assert seen == ["body", "scaled"]


def test_cancel_before_a_write_leaves_nothing_written():
    """The property that makes cancellation safe rather than merely fast."""
    storage = _storage()
    written_before = set(storage.keys())

    with pytest.raises(RunCancelled) as excinfo:
        run_template(
            _template(),
            storage=storage,
            # Let everything compute, then refuse at the write boundary.
            cancel=lambda: True,
        )

    assert excinfo.value.phase == "load"
    assert set(storage.keys()) == written_before


def test_cancel_is_polled_between_windows():
    calls = {"n": 0}

    def cancel() -> bool:
        calls["n"] += 1
        return calls["n"] > 3

    with pytest.raises(RunCancelled):
        run_template(_template(), storage=_storage(), chunk_size=8, cancel=cancel)


def test_cancelled_is_a_cfdmod_error_not_a_bare_runtime_error():
    """A service must be able to tell "user cancelled" from "run failed"."""
    from cfdmod.core.errors import CfdmodError

    assert issubclass(RunCancelled, CfdmodError)


def test_no_callbacks_is_the_unchanged_path():
    # Intermediates are released once nothing downstream reads them, so the
    # notebook view needs return_all -- see test_run_template_outputs.py.
    result = run_template(_template(), storage=_storage(), return_all=True)
    assert set(result) >= {"body", "scaled", "halved"}


def test_progress_callback_raising_is_not_swallowed_as_cancellation():
    """An exception in the caller's logging is a bug, not an abort."""

    def boom(event: RunEvent) -> None:
        raise KeyError("logging blew up")

    with pytest.raises(KeyError):
        run_template(_template(), storage=_storage(), on_progress=boom)


@pytest.mark.parametrize(("index", "total", "expected"), [(0, 0, 1.0), (5, 3, 1.0), (0, 4, 0.25)])
def test_fraction_is_bounded(index, total, expected):
    assert RunEvent(phase="step", name="x", index=index, total=total).fraction == expected
