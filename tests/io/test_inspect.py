"""Tests for cfdmod.io.inspect (debug helpers for HDF5 files on disk)."""

from __future__ import annotations

import h5py
import numpy as np
import pytest

from cfdmod.io.inspect import inspect_h5, read_all_timesteps


@pytest.fixture
def sample_h5(tmp_path):
    """Build a small H5 mimicking the pipeline's timeseries layout.

    Layout:
        /Triangles            (4, 3) int32
        /Geometry             (12, 3) float64
        /cp/t0.0              (4,) float32
        /cp/t0.1              (4,) float32
        /cp/t0.2              (4,) float32
        /meta/time_steps      (3,) float64  -> [0, 0.1, 0.2]
        /meta/time_normalized (3,) float64
        /processing_metadata  group with one attribute
    """
    path = tmp_path / "sample.h5"
    with h5py.File(path, "w") as f:
        f.create_dataset("Triangles", data=np.zeros((4, 3), dtype=np.int32))
        f.create_dataset("Geometry", data=np.zeros((12, 3), dtype=np.float64))
        cp = f.create_group("cp")
        cp.create_dataset("t0.0", data=np.array([1, 2, 3, 4], dtype=np.float32))
        cp.create_dataset("t0.1", data=np.array([5, 6, 7, 8], dtype=np.float32))
        cp.create_dataset("t0.2", data=np.array([9, 10, 11, 12], dtype=np.float32))
        meta = f.create_group("meta")
        meta.create_dataset("time_steps", data=np.array([0.0, 0.1, 0.2], dtype=np.float64))
        meta.create_dataset(
            "time_normalized", data=np.array([0.0, 0.5, 1.0], dtype=np.float64)
        )
        pm = f.create_group("processing_metadata")
        pm.attrs["config"] = "{kind: cp, body: galpao}"
    return path


def test_inspect_h5_prints_header_and_top_level_datasets(sample_h5, capsys):
    inspect_h5(sample_h5)
    out = capsys.readouterr().out
    assert "sample.h5" in out
    assert "Triangles" in out
    assert "(4, 3)" in out
    assert "int32" in out
    assert "Geometry" in out
    assert "float64" in out


def test_inspect_h5_collapses_timestep_group(sample_h5, capsys):
    inspect_h5(sample_h5)
    out = capsys.readouterr().out
    # The cp/ group should appear as a one-line summary, not three
    # individual t0.0 / t0.1 / t0.2 lines.
    assert "cp/" in out
    assert "3 timesteps" in out
    assert "step shape=(4,)" in out
    assert "t0.0" not in out  # not expanded
    assert "t0.1" not in out


def test_inspect_h5_shows_meta_range_preview(sample_h5, capsys):
    inspect_h5(sample_h5)
    out = capsys.readouterr().out
    assert "time_steps" in out
    # Range preview for a /meta dataset.
    assert "range=[0 .. 0.2]" in out


def test_inspect_h5_can_suppress_attrs_and_meta_preview(sample_h5, capsys):
    inspect_h5(sample_h5, show_attrs=False, show_meta_values=False)
    out = capsys.readouterr().out
    assert "@config" not in out  # attribute hidden
    assert "range=[" not in out  # meta preview hidden


def test_inspect_h5_shows_attrs_when_enabled(sample_h5, capsys):
    inspect_h5(sample_h5, show_attrs=True)
    out = capsys.readouterr().out
    assert "@config" in out
    assert "galpao" in out


def test_read_all_timesteps_sorts_by_time_and_stacks(sample_h5):
    times, data = read_all_timesteps(sample_h5, "cp")
    assert times.dtype == np.float64
    np.testing.assert_array_equal(times, np.array([0.0, 0.1, 0.2]))
    assert data.shape == (3, 4)
    assert data.dtype == np.float32
    np.testing.assert_array_equal(data[0], [1, 2, 3, 4])
    np.testing.assert_array_equal(data[2], [9, 10, 11, 12])


def test_read_all_timesteps_missing_group_raises(sample_h5):
    with pytest.raises(KeyError, match="not found"):
        read_all_timesteps(sample_h5, "does_not_exist")


def test_read_all_timesteps_group_with_no_time_keys_raises(tmp_path):
    path = tmp_path / "bad.h5"
    with h5py.File(path, "w") as f:
        g = f.create_group("not_timeseries")
        g.create_dataset("foo", data=np.array([1, 2, 3]))
    with pytest.raises(ValueError, match="no t"):
        read_all_timesteps(path, "not_timeseries")


def test_read_all_timesteps_handles_unsorted_keys(tmp_path):
    """Time keys are not guaranteed insertion-sorted on disk; the reader
    must still return them in ascending time order."""
    path = tmp_path / "shuffled.h5"
    with h5py.File(path, "w") as f:
        g = f.create_group("v")
        # Insert deliberately out of order.
        g.create_dataset("t2.0", data=np.array([20.0]))
        g.create_dataset("t0.5", data=np.array([5.0]))
        g.create_dataset("t1.0", data=np.array([10.0]))
    times, data = read_all_timesteps(path, "v")
    np.testing.assert_array_equal(times, [0.5, 1.0, 2.0])
    np.testing.assert_array_equal(data.flatten(), [5.0, 10.0, 20.0])


def test_inspect_h5_works_on_file_with_only_groups(tmp_path, capsys):
    path = tmp_path / "groups_only.h5"
    with h5py.File(path, "w") as f:
        f.create_group("a")
        f.create_group("b")
    inspect_h5(path)
    out = capsys.readouterr().out
    assert "a/" in out
    assert "b/" in out


def test_top_level_imports_are_reachable():
    """Both helpers must be importable from the top-level cfdmod package."""
    from cfdmod import inspect_h5 as top_inspect
    from cfdmod import read_all_timesteps as top_read
    from cfdmod.io import inspect_h5 as io_inspect
    from cfdmod.io import read_all_timesteps as io_read

    assert top_inspect is io_inspect
    assert top_read is io_read
