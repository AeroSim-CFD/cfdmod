"""Unit tests for :class:`cfdmod.core.time_axis.TimeAxis`."""

from __future__ import annotations

import numpy as np
import pytest

from cfdmod.core import TimeAxis


def test_times_and_normalised_match_definition():
    axis = TimeAxis(initial_time=10.0, timestep_size=0.5, n_timesteps=4)
    assert np.allclose(axis.times(), [10.0, 10.5, 11.0, 11.5])
    # Default normalization offset is initial_time -> times_normalized[0] == 0
    assert np.allclose(axis.times_normalized(), [0.0, 0.5, 1.0, 1.5])


def test_with_normalization_offset_overrides_default():
    axis = TimeAxis(initial_time=10.0, timestep_size=0.5, n_timesteps=4)
    shifted = axis.with_normalization_offset(8.0)
    assert np.allclose(shifted.times_normalized(), [2.0, 2.5, 3.0, 3.5])


def test_window_returns_subaxis_and_slice():
    axis = TimeAxis(initial_time=0.0, timestep_size=0.1, n_timesteps=10)
    sub_axis, sl = axis.window(0.3, 0.6)
    assert sl == slice(3, 7)
    assert sub_axis.n_timesteps == 4
    assert np.isclose(sub_axis.initial_time, 0.3)
    assert np.isclose(sub_axis.timestep_size, 0.1)


def test_translate_and_rescale():
    axis = TimeAxis(initial_time=1.0, timestep_size=0.25, n_timesteps=4)
    t = axis.translate(5.0)
    assert t.initial_time == 5.0 and t.timestep_size == 0.25 and t.n_timesteps == 4
    r = axis.rescale(2.0)
    assert r.initial_time == 2.0 and r.timestep_size == 0.5


def test_time_aggregated_axis_blocks_window_and_index_for_time():
    axis = TimeAxis(initial_time=0.0, timestep_size=0.0, n_timesteps=0)
    assert axis.is_time_aggregated
    with pytest.raises(ValueError):
        axis.window(0.0, 1.0)
    with pytest.raises(ValueError):
        axis.index_for_time(0.0)


def test_index_for_time_clamps_to_range():
    axis = TimeAxis(initial_time=0.0, timestep_size=1.0, n_timesteps=5)
    assert axis.index_for_time(-100.0) == 0
    assert axis.index_for_time(100.0) == 4
    assert axis.index_for_time(2.4) == 2
    assert axis.index_for_time(2.6) == 3


def test_zero_timestep_with_positive_n_timesteps_rejected():
    with pytest.raises(ValueError):
        TimeAxis(initial_time=0.0, timestep_size=0.0, n_timesteps=4)


def test_time_at_negative_index_counts_from_end():
    axis = TimeAxis(initial_time=10.0, timestep_size=1.0, n_timesteps=5)
    assert axis.time_at(-1) == 14.0
    assert axis.time_at(-5) == 10.0
    with pytest.raises(IndexError):
        axis.time_at(-6)
