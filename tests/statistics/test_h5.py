"""Tests for cfdmod.statistics.apply_statistics_h5 (file-in wrapper)."""

from __future__ import annotations

import pathlib

import numpy as np
import pytest

from cfdmod.io.xdmf import (
    write_temporal_xdmf,
    write_timeseries_geometry,
    write_timeseries_meta,
    write_timeseries_step,
)
from cfdmod.statistics import (
    BasicStatisticModel,
    ExtremeAbsoluteParamsModel,
    ParameterizedStatisticModel,
    apply_statistics_h5,
)

pytestmark = pytest.mark.unit


def _build_h5(tmp_path: pathlib.Path, n_t: int = 60, n_pts: int = 4) -> pathlib.Path:
    """Synth a tiny timeseries H5 with cp/t{T} datasets."""
    h5 = tmp_path / "cp.h5"
    write_timeseries_geometry(h5, np.zeros((n_pts, 3), dtype=np.int32), np.zeros((n_pts * 3, 3)))
    t = np.linspace(0.0, 5.9, n_t)
    rng = np.random.default_rng(seed=1)
    signal = np.sin(2 * np.pi * t)[:, None] + rng.normal(0.0, 0.3, (n_t, n_pts))
    for i, ti in enumerate(t):
        write_timeseries_step(h5, "cp", f"t{ti}", signal[i])
    write_timeseries_meta(h5, t, t)
    write_temporal_xdmf(h5, h5.with_suffix(".xdmf"), "cp")
    return h5, t, signal


def test_streaming_path_for_basic_moments(tmp_path):
    h5, _, signal = _build_h5(tmp_path)
    stats = [BasicStatisticModel(stats="mean"), BasicStatisticModel(stats="rms")]
    df = apply_statistics_h5(h5_path=h5, group="cp", statistics=stats)
    assert "mean" in df.columns and "rms" in df.columns
    np.testing.assert_allclose(df["mean"].values, signal.mean(axis=0), rtol=1e-10)
    # Welford uses ddof=1, matches np.std(..., ddof=1).
    np.testing.assert_allclose(df["rms"].values, signal.std(axis=0, ddof=1), rtol=1e-10)


def test_full_load_path_for_extreme_methods(tmp_path):
    h5, _, signal = _build_h5(tmp_path)
    stats = [
        ParameterizedStatisticModel(stats="min", params=ExtremeAbsoluteParamsModel()),
        ParameterizedStatisticModel(stats="max", params=ExtremeAbsoluteParamsModel()),
    ]
    df = apply_statistics_h5(h5_path=h5, group="cp", statistics=stats)
    np.testing.assert_allclose(df["min"].values, signal.min(axis=0))
    np.testing.assert_allclose(df["max"].values, signal.max(axis=0))


def test_timestep_range_filters_keys(tmp_path):
    h5, t, signal = _build_h5(tmp_path)
    stats = [BasicStatisticModel(stats="mean")]
    full = apply_statistics_h5(h5_path=h5, group="cp", statistics=stats)
    # Restrict to the first half of the time axis.
    half = apply_statistics_h5(
        h5_path=h5,
        group="cp",
        statistics=stats,
        timestep_range=(t[0], t[len(t) // 2]),
    )
    # Same number of points, different mean value.
    assert len(full) == len(half)
    assert not np.allclose(full["mean"].values, half["mean"].values)


def test_empty_range_raises(tmp_path):
    h5, _, _ = _build_h5(tmp_path)
    with pytest.raises(ValueError, match="No keys found"):
        apply_statistics_h5(
            h5_path=h5,
            group="cp",
            statistics=[BasicStatisticModel(stats="mean")],
            timestep_range=(1000.0, 2000.0),
        )
