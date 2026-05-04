"""Tests for cfdmod.filters.apply_filters_h5 (file-in / file-out wrapper)."""

from __future__ import annotations

import pathlib

import h5py
import numpy as np
import pytest

from cfdmod.filters import MovingAverageFilter, apply_filters_h5
from cfdmod.io.xdmf import (
    read_processing_metadata,
    write_temporal_xdmf,
    write_timeseries_geometry,
    write_timeseries_meta,
    write_timeseries_step,
)

pytestmark = pytest.mark.unit


def _build_cp(tmp_path: pathlib.Path, n_t: int = 100, n_tri: int = 5) -> pathlib.Path:
    """Synth a tiny cp.h5 with `sin + noise` per triangle."""
    h5 = tmp_path / "cp.h5"
    triangles = np.tile(np.array([[0, 1, 2]]), (n_tri, 1))
    vertices = np.zeros((10, 3))
    write_timeseries_geometry(h5, triangles, vertices)
    t = np.linspace(0.0, 10.0, n_t)
    rng = np.random.default_rng(seed=0)
    signal = np.sin(2 * np.pi * t)[:, None] + rng.normal(0.0, 0.5, (n_t, n_tri))
    for i, ti in enumerate(t):
        write_timeseries_step(h5, "cp", f"t{ti}", signal[i])
    write_timeseries_meta(h5, t, t)
    write_temporal_xdmf(h5, h5.with_suffix(".xdmf"), "cp")
    return h5


def test_apply_filters_h5_reduces_noise(tmp_path):
    """Moving average with a non-trivial window should cut std significantly
    while leaving the per-triangle shape intact."""
    src = _build_cp(tmp_path)
    out = tmp_path / "cp.smoothed.h5"
    apply_filters_h5(src, out, filters=[MovingAverageFilter(window=1.0)], group="cp")

    with h5py.File(out, "r") as f:
        keys = sorted(f["cp"].keys(), key=lambda k: float(k[1:]))
        filtered = np.stack([f["cp"][k][:] for k in keys])
        assert "Triangles" in f and "Geometry" in f
        assert "meta" in f and "time_steps" in f["meta"]

    assert filtered.shape == (100, 5)
    with h5py.File(src, "r") as f:
        keys = sorted(f["cp"].keys(), key=lambda k: float(k[1:]))
        original = np.stack([f["cp"][k][:] for k in keys])
    assert filtered.std() < 0.5 * original.std()


def test_apply_filters_h5_records_metadata(tmp_path):
    src = _build_cp(tmp_path)
    out = tmp_path / "cp.smoothed.h5"
    chain = [MovingAverageFilter(window=2.0)]
    apply_filters_h5(src, out, filters=chain, group="cp")

    meta = read_processing_metadata(out, "/")
    assert meta["config"] == {"filters": [{"kind": "moving_average", "window": 2.0}]}
    assert meta["source_h5"] == str(src)
    assert meta["group"] == "cp"


def test_apply_filters_h5_chain_order_matters(tmp_path):
    """Two MA filters back-to-back smooth more than one alone."""
    src = _build_cp(tmp_path)
    out_one = tmp_path / "one.h5"
    out_two = tmp_path / "two.h5"

    apply_filters_h5(
        src, out_one, filters=[MovingAverageFilter(window=0.5)], group="cp"
    )
    apply_filters_h5(
        src,
        out_two,
        filters=[MovingAverageFilter(window=0.5), MovingAverageFilter(window=0.5)],
        group="cp",
    )

    def load(h5):
        with h5py.File(h5, "r") as f:
            keys = sorted(f["cp"].keys(), key=lambda k: float(k[1:]))
            return np.stack([f["cp"][k][:] for k in keys])

    assert load(out_two).std() < load(out_one).std()


def test_apply_filters_h5_empty_chain_errors(tmp_path):
    src = _build_cp(tmp_path)
    with pytest.raises(ValueError, match="filters list is empty"):
        apply_filters_h5(src, tmp_path / "out.h5", filters=[], group="cp")


def test_apply_filters_h5_rejects_non_uniform_dt(tmp_path):
    """If the input timesteps are not uniformly spaced, the wrapper refuses."""
    h5 = tmp_path / "cp.h5"
    triangles = np.array([[0, 1, 2]])
    vertices = np.zeros((3, 3))
    write_timeseries_geometry(h5, triangles, vertices)
    t = np.array([0.0, 0.1, 0.5, 0.6, 1.0])
    rng = np.random.default_rng(0)
    for ti in t:
        write_timeseries_step(h5, "cp", f"t{ti}", rng.normal(size=1))
    write_timeseries_meta(h5, t, t)

    with pytest.raises(ValueError, match="uniform timestep spacing"):
        apply_filters_h5(
            h5,
            tmp_path / "out.h5",
            filters=[MovingAverageFilter(window=0.5)],
            group="cp",
        )
