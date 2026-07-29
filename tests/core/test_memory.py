"""The budget arithmetic that turns RAM into a time-chunk size."""

from __future__ import annotations

import numpy as np
import pytest

from cfdmod.core.memory import (
    ChunkPlan,
    bytes_per_timestep,
    estimate_peak_bytes,
    plan_chunking,
    suggest_chunk_size,
)

pytestmark = pytest.mark.unit


def test_bytes_per_timestep_counts_every_live_array():
    # 1000 elements x 4 bytes (float32) x 3 live arrays
    assert bytes_per_timestep(1000, n_live_arrays=3, dtype=np.float32) == 12_000


def test_bytes_per_timestep_follows_the_dtype():
    assert bytes_per_timestep(1000, dtype=np.float64) == 8_000
    assert bytes_per_timestep(1000, dtype=np.float32) == 4_000


def test_estimate_peak_is_the_unit_times_the_window():
    assert estimate_peak_bytes(1000, 50, n_live_arrays=2, dtype=np.float32) == 400_000


def test_suggest_chunk_size_spends_the_budget_minus_headroom():
    # unit = 1000 * 4 * 2 = 8000 B/step; 8 MB * 0.8 safety / 8000 = 800 steps
    assert (
        suggest_chunk_size(1000, 10_000, budget_bytes=8_000_000, n_live_arrays=2, dtype=np.float32)
        == 800
    )


def test_suggest_chunk_size_never_exceeds_the_series():
    assert suggest_chunk_size(10, 25, budget_bytes=10**9, n_live_arrays=1) == 25


def test_suggest_chunk_size_never_returns_zero():
    """A window of zero steps processes nothing; one step over budget is the
    honest answer, and the plan says so."""
    plan = plan_chunking(10**7, 1000, budget_bytes=1000, n_live_arrays=4)
    assert plan.chunk_size == 1
    assert plan.clamped
    assert plan.exceeds_budget


def test_plan_without_budget_or_chunk_is_a_single_pass():
    plan = plan_chunking(500, 200, n_live_arrays=2)
    assert plan.chunk_size == 200
    assert not plan.is_chunked
    # Still prices the run -- that number is what says whether to chunk at all.
    assert plan.estimated_peak_bytes == 500 * 4 * 2 * 200


def test_plan_with_explicit_chunk_size_is_clamped_to_the_series():
    plan = plan_chunking(10, 30, chunk_size=100)
    assert plan.chunk_size == 30
    assert plan.clamped
    assert not plan.is_chunked


def test_plan_rejects_both_knobs():
    with pytest.raises(ValueError, match="not both"):
        plan_chunking(10, 30, budget_bytes=1000, chunk_size=5)


@pytest.mark.parametrize(
    ("kwargs", "match"),
    [
        ({"budget_bytes": 0}, "budget_bytes must be positive"),
        ({"chunk_size": 0}, "chunk_size must be positive"),
        ({"n_live_arrays": 0}, "n_live_arrays must be at least 1"),
        ({"safety_factor": 0}, "safety_factor must be in"),
        ({"safety_factor": 1.5}, "safety_factor must be in"),
    ],
)
def test_plan_rejects_degenerate_inputs(kwargs, match):
    with pytest.raises(ValueError, match=match):
        plan_chunking(10, 30, **kwargs)


def test_zero_elements_does_not_divide_by_zero():
    plan = plan_chunking(0, 100, budget_bytes=1000)
    assert plan.chunk_size == 100
    assert plan.estimated_peak_bytes == 0


def test_describe_says_off_when_not_chunked():
    assert "off" in plan_chunking(100, 50).describe()


def test_describe_reports_windows_and_the_budget_overrun():
    plan = plan_chunking(1000, 100, chunk_size=30, n_live_arrays=2)
    text = plan.describe()
    assert "30 steps x 4 windows" in text

    tight = plan_chunking(10**7, 1000, budget_bytes=1000, n_live_arrays=4)
    assert "BUDGET EXCEEDED" in tight.describe()


def test_chunk_plan_is_frozen():
    plan = plan_chunking(10, 10)
    assert isinstance(plan, ChunkPlan)
    with pytest.raises(Exception):
        plan.chunk_size = 5  # type: ignore[misc]
