"""Tests for cfdmod.io.timeseries (read_timeseries_df / to_csv / plot)."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from cfdmod.io.timeseries import plot_timeseries, read_timeseries_df, to_csv
from cfdmod.io.xdmf import write_timeseries_geometry, write_timeseries_meta, write_timeseries_step

pytestmark = pytest.mark.unit


@pytest.fixture()
def per_triangle_h5(tmp_path):
    """Tiny per-triangle Cp-style file: 4 timesteps x 3 distinct values per tri."""
    path = tmp_path / "ts.h5"
    triangles = np.array([[0, 1, 2], [1, 3, 2]], dtype=np.int32)
    vertices = np.array([[0, 0, 0], [0, 1, 0], [1, 0, 0], [1, 1, 0]], dtype=np.float64)
    write_timeseries_geometry(path, triangles, vertices)
    times = np.array([0.0, 1.0, 2.0, 3.0])
    for i, t in enumerate(times):
        write_timeseries_step(path, "cp", f"t{t}", np.array([1.0 + i, 2.0 + i, 3.0 + i]))
    write_timeseries_meta(path, time_steps=times, time_normalized=times / 10.0)
    return path


@pytest.fixture()
def per_region_h5(tmp_path):
    """6-triangle Cf-style file: triangles 0..2 in region A (constant value),
    3..5 in region B (different constant)."""
    path = tmp_path / "ts_regions.h5"
    triangles = np.zeros((6, 3), dtype=np.int32)
    vertices = np.zeros((6, 3), dtype=np.float64)
    write_timeseries_geometry(path, triangles, vertices)
    times = np.array([0.0, 0.5, 1.0])
    region_a = [0.1, 0.2, 0.3]
    region_b = [-0.5, -0.4, -0.3]
    for ra, rb, t in zip(region_a, region_b, times):
        write_timeseries_step(path, "cf_x", f"t{t}", np.array([ra, ra, ra, rb, rb, rb]))
    write_timeseries_meta(path, time_steps=times, time_normalized=times)
    return path


def test_read_returns_wide_dataframe(per_triangle_h5):
    df = read_timeseries_df(per_triangle_h5, "cp")
    assert df.index.name == "time_normalized"
    assert list(df.index) == [0.0, 0.1, 0.2, 0.3]  # raw / 10
    assert list(df.columns) == [0, 1, 2]
    assert df.shape == (4, 3)


def test_filter_by_triangles(per_triangle_h5):
    df = read_timeseries_df(per_triangle_h5, "cp", triangles=[0, 2])
    assert list(df.columns) == [0, 2]
    np.testing.assert_array_equal(df[0].to_numpy(), [1.0, 2.0, 3.0, 4.0])
    np.testing.assert_array_equal(df[2].to_numpy(), [3.0, 4.0, 5.0, 6.0])


def test_filter_by_timestep_range_inclusive(per_triangle_h5):
    df = read_timeseries_df(per_triangle_h5, "cp", timestep_range=(1.0, 2.0))
    assert df.shape == (2, 3)
    np.testing.assert_array_equal(df.index.to_numpy(), [0.1, 0.2])


def test_regions_dedupe_collapses_constant_columns(per_region_h5):
    df = read_timeseries_df(per_region_h5, "cf_x", regions=True)
    assert df.shape == (3, 2)
    # Representatives: lowest tri index per unique value -> 0 and 3.
    assert list(df.columns) == [0, 3]
    np.testing.assert_array_equal(df[0].to_numpy(), [0.1, 0.2, 0.3])
    np.testing.assert_array_equal(df[3].to_numpy(), [-0.5, -0.4, -0.3])


def test_regions_and_triangles_are_mutually_exclusive(per_region_h5):
    with pytest.raises(ValueError, match="not both"):
        read_timeseries_df(per_region_h5, "cf_x", regions=True, triangles=[0])


def test_max_columns_guard_for_wide_files(tmp_path):
    """A per-triangle file with > max_columns columns must refuse without
    a triangles filter."""
    path = tmp_path / "wide.h5"
    n_tri = 50
    write_timeseries_geometry(path, np.zeros((n_tri, 3), dtype=np.int32), np.zeros((1, 3)))
    write_timeseries_step(path, "cp", "t0.0", np.arange(n_tri, dtype=np.float64))
    write_timeseries_meta(path, time_steps=np.array([0.0]), time_normalized=np.array([0.0]))

    with pytest.raises(ValueError, match="too wide"):
        read_timeseries_df(path, "cp", max_columns=10)

    df = read_timeseries_df(path, "cp", max_columns=10, triangles=[0, 5, 9])
    assert list(df.columns) == [0, 5, 9]


def test_missing_group(per_triangle_h5):
    with pytest.raises(ValueError, match="not found"):
        read_timeseries_df(per_triangle_h5, "cp_missing")


def test_to_csv_roundtrip(per_region_h5, tmp_path):
    df = read_timeseries_df(per_region_h5, "cf_x", regions=True)
    csv_path = tmp_path / "cf_x.csv"
    to_csv(df, csv_path)

    reread = pd.read_csv(csv_path, index_col=0)
    reread.index.name = "time_normalized"
    np.testing.assert_array_equal(reread.to_numpy(), df.to_numpy())
    assert list(int(c) for c in reread.columns) == list(df.columns)


def test_plot_timeseries_returns_axes(per_region_h5):
    pytest.importorskip("matplotlib")
    df = read_timeseries_df(per_region_h5, "cf_x", regions=True)
    ax = plot_timeseries(df, title="Cf_x demo", ylabel="Cf_x")
    assert ax.get_xlabel() == "time_normalized"
    assert ax.get_ylabel() == "Cf_x"
    assert ax.get_title() == "Cf_x demo"
