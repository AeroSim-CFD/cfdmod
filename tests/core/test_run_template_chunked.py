"""Running a template over time windows: same numbers, smaller peak.

`cfdmod.core.chunked` has been able to stream a pipeline over the time axis
since it was written, and nothing called it - so `run_template`, the path a
service drives, always materialised the full ``(n_elements, n_timesteps)``
array for every intermediate. These tests cover the wiring, and the two
properties that make it worth having: the outputs must not move, and the peak
must fall.
"""

from __future__ import annotations

import tracemalloc

import numpy as np
import pytest

from cfdmod.adapters.memory import MemoryFieldStore, MemoryStorage
from cfdmod.core import ElementMeta, Grouping, SurfaceDataSource, TimeAxis, Topology
from cfdmod.core.errors import TemplateError
from cfdmod.core.memory import ChunkPlan
from cfdmod.core.pipeline_yaml import PipelineTemplate, run_template

pytestmark = pytest.mark.unit


def _surface(n_elements: int, n_timesteps: int, seed: int = 0) -> SurfaceDataSource:
    rng = np.random.default_rng(seed)
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


def _scaling_template(name: str = "chunkable") -> PipelineTemplate:
    """A time-length-preserving chain: every op declares time chunkability."""
    return PipelineTemplate.model_validate(
        {
            "name": name,
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
                    "id": "smoothed",
                    "kind": "scale",
                    "source": "scaled",
                    "field": "cp",
                    "factor": 0.5,
                    "out": "cp_half",
                },
            ],
            "outputs": {},
        }
    )


def _storage_with(ds: SurfaceDataSource) -> MemoryStorage:
    storage = MemoryStorage()
    storage.write_data_source("body", ds)
    return storage


def test_chunked_and_unchunked_agree_exactly():
    """The whole point: chunking changes the peak, not the answer."""
    ds = _surface(40, 96)
    template = _scaling_template()

    whole = run_template(template, storage=_storage_with(ds))
    windowed = run_template(template, storage=_storage_with(ds), chunk_size=10)

    for name, field in (("scaled", "cp"), ("smoothed", "cp_half")):
        np.testing.assert_array_equal(
            windowed[name].fields.read(field), whole[name].fields.read(field)
        )
        assert windowed[name].time.n_timesteps == whole[name].time.n_timesteps
        np.testing.assert_allclose(windowed[name].time.times(), whole[name].time.times())


def test_uneven_final_window_is_handled():
    """96 is not a multiple of 7; the short last window must not truncate."""
    ds = _surface(12, 96)
    template = _scaling_template()

    whole = run_template(template, storage=_storage_with(ds))
    windowed = run_template(template, storage=_storage_with(ds), chunk_size=7)

    assert windowed["smoothed"].time.n_timesteps == 96
    np.testing.assert_array_equal(
        windowed["smoothed"].fields.read("cp_half"), whole["smoothed"].fields.read("cp_half")
    )


def test_memory_budget_derives_the_window_and_reports_it():
    ds = _surface(1000, 200)
    plans: list[ChunkPlan] = []

    run_template(
        _scaling_template(),
        storage=_storage_with(ds),
        memory_budget=1_000_000,
        on_plan=plans.append,
    )

    (plan,) = plans
    assert plan.is_chunked
    # 3 live arrays (1 input + 2 steps) x 1000 elements x 4 bytes = 12 kB/step;
    # 1 MB x 0.8 / 12 kB = 66 steps.
    assert plan.n_live_arrays == 3
    assert plan.chunk_size == 66
    assert "66 steps" in plan.describe()


def test_on_plan_fires_even_when_not_chunking():
    """A caller logging the plan should not have to special-case the off path."""
    plans: list[ChunkPlan] = []
    run_template(_scaling_template(), storage=_storage_with(_surface(8, 16)), on_plan=plans.append)

    (plan,) = plans
    assert not plan.is_chunked
    assert plan.estimated_peak_bytes > 0


def test_statistics_cannot_be_chunked_and_says_so():
    """Windowed statistics are not the statistics of the series.

    `statistics` declares chunkability along elements, not time, so asking for
    a chunked run must fail loudly rather than return plausible wrong numbers.
    """
    template = PipelineTemplate.model_validate(
        {
            "name": "with_stats",
            "inputs": {"body": {"kind": "surface", "path": "body", "field": "pressure"}},
            "pipeline": [
                {
                    "id": "stats",
                    "kind": "statistics",
                    "source": "body",
                    "field": "pressure",
                    "kinds": ["mean", "rms"],
                },
            ],
            "outputs": {},
        }
    )
    with pytest.raises(TemplateError, match="cannot run this template chunked"):
        run_template(template, storage=_storage_with(_surface(8, 64)), chunk_size=8)


def test_unchunkable_template_is_rejected_before_any_step_runs():
    """Fail on the contract, not halfway through the data."""
    calls: list[str] = []

    template = PipelineTemplate.model_validate(
        {
            "name": "with_stats",
            "inputs": {"body": {"kind": "surface", "path": "body", "field": "pressure"}},
            "pipeline": [
                {
                    "id": "stats",
                    "kind": "statistics",
                    "source": "body",
                    "field": "pressure",
                    "kinds": ["mean"],
                },
            ],
            "outputs": {},
        }
    )

    class WatchingStorage(MemoryStorage):
        def read_data_source(self, key, *, kind=None):
            calls.append(key)
            return super().read_data_source(key, kind=kind)

    storage = WatchingStorage()
    storage.write_data_source("body", _surface(8, 64))
    with pytest.raises(TemplateError):
        run_template(template, storage=storage, chunk_size=8)
    # The input is read once to size the plan; no step ran and nothing was written.
    assert calls == ["body"]


def test_single_timestep_source_is_never_chunked():
    ds = _surface(8, 1)
    plans: list[ChunkPlan] = []
    run_template(
        _scaling_template(), storage=_storage_with(ds), chunk_size=1, on_plan=plans.append
    )
    assert not plans[0].is_chunked


def test_chunking_lowers_the_peak_allocation():
    """The property the whole feature exists for, measured rather than assumed.

    Reducing the element axis is what makes the win show up - here a
    ``field_series_for_groups`` collapses 4000 triangles onto 4 groups, so the
    big per-triangle intermediates live for one window at a time while the
    concatenated result stays small.
    """
    n_elements, n_timesteps, n_groups = 20_000, 600, 4
    rng = np.random.default_rng(3)
    verts = rng.random((n_elements * 3, 3))
    tris = np.arange(n_elements * 3, dtype=np.int32).reshape(n_elements, 3)
    # Attached up front rather than built by a grouping op, so the template
    # stays about the chunking rather than about mesh ingest.
    grouping = Grouping(
        name="floors",
        indices=np.arange(n_elements) % n_groups,
        labels=[f"g{i}" for i in range(n_groups)],
    )
    ds = SurfaceDataSource(
        time=TimeAxis(initial_time=0.0, timestep_size=0.1, n_timesteps=n_timesteps),
        topology=Topology.triangles(tris, verts),
        elements=ElementMeta(area=np.ones(n_elements)),
        fields=MemoryFieldStore(
            {"pressure": rng.random((n_elements, n_timesteps)).astype(np.float32)}
        ),
    ).with_grouping(grouping)
    template = PipelineTemplate.model_validate(
        {
            "name": "reducing",
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
                    "id": "per_group",
                    "kind": "field_series_for_groups",
                    "source": "scaled",
                    "grouping": "floors",
                    "field": "cp",
                    "agg": "area_weighted_mean",
                    "out": "cp_group",
                },
            ],
            # Declaring the output is what lets the chunked run drop the
            # per-triangle intermediate instead of accumulating every window
            # of it -- see _retained_bindings.
            "outputs": {"per_floor": {"source": "per_group", "path": "out"}},
        }
    )

    def peak_bytes(**kwargs):
        tracemalloc.start()
        try:
            result = run_template(template, storage=_storage_with(ds), **kwargs)
            peak = tracemalloc.get_traced_memory()[1]
        finally:
            tracemalloc.stop()
        return peak, result

    whole_peak, whole = peak_bytes()
    windowed_peak, windowed = peak_bytes(chunk_size=50)

    np.testing.assert_allclose(
        windowed["per_group"].fields.read("cp_group"),
        whole["per_group"].fields.read("cp_group"),
        rtol=1e-6,
    )
    # A bare "lower" would pass on noise. Two claims instead: the chunked run
    # must be a clear fraction of the unchunked one, and it must stay under the
    # size of the field itself -- the array an unchunked run has no choice but
    # to hold whole. Both are loose against the ~7x measured at this size.
    field_bytes = n_elements * n_timesteps * 4
    assert windowed_peak < whole_peak / 2, (
        f"chunked peak {windowed_peak / 1e6:.1f} MB was not well below "
        f"unchunked {whole_peak / 1e6:.1f} MB"
    )
    assert windowed_peak < field_bytes, (
        f"chunked peak {windowed_peak / 1e6:.1f} MB exceeded the whole field "
        f"({field_bytes / 1e6:.1f} MB)"
    )
