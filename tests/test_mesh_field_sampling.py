"""Tests for the mesh_field sampling + time-window stats helpers."""

from __future__ import annotations

import types

import numpy as np
import pytest

from cfdmod import mesh_field

pytestmark = pytest.mark.unit


def _geometry(triangle_vertices: np.ndarray):
    """Minimal LnasGeometry stand-in: only .triangle_vertices is used."""
    return types.SimpleNamespace(triangle_vertices=triangle_vertices)


def _three_tris_along_x():
    # Three tiny triangles with centroids at x = 0, 1, 2 (y=z=0).
    def tri(cx):
        return [[cx - 0.1, 0.0, 0.0], [cx + 0.1, 0.0, 0.0], [cx, 0.1, 0.0]]

    return np.array([tri(0.0), tri(1.0), tri(2.0)], dtype=float)


def test_triangle_centroids():
    geom = _geometry(_three_tris_along_x())
    c = mesh_field.triangle_centroids(geom)
    assert c.shape == (3, 3)
    assert np.allclose(c[:, 0], [0.0, 1.0, 2.0], atol=1e-9)


def test_sample_field_along_line_picks_nearest_triangle():
    geom = _geometry(_three_tris_along_x())
    field = np.array([10.0, 20.0, 30.0])
    df = mesh_field.sample_field_along_line(geom, field, (0.0, 0.0, 0.0), (2.0, 0.0, 0.0), n=5)
    assert list(df.columns) == ["s", "x", "y", "z", "value"]
    assert df.shape[0] == 5
    # endpoints hit tri0 (10) and tri2 (30); midpoint hits tri1 (20)
    assert df["value"].iloc[0] == 10.0
    assert df["value"].iloc[-1] == 30.0
    assert df["value"].iloc[2] == 20.0
    assert df["s"].iloc[-1] == pytest.approx(2.0)


def test_moving_average_stats_smooths_and_peaks():
    # A spike smoothed over a 3-sample window has a lower peak than the raw max.
    series = np.array([0.0, 0.0, 9.0, 0.0, 0.0])
    out = mesh_field.moving_average_stats(series, dt=1.0, window_s=3.0)
    assert out["window"] == 3
    assert out["mean"] == pytest.approx(1.8)
    assert out["ma_max"] == pytest.approx(3.0)  # (0+9+0)/3
    assert out["ma_max"] < series.max()


def test_moving_average_stats_window_rounds_odd_and_stays_aligned():
    # window_s/dt = 3 (odd) -> window 3; edge-padded output keeps the input length.
    out = mesh_field.moving_average_stats(
        np.array([0.0, 0.0, 9.0, 0.0, 0.0]), dt=1.0, window_s=3.0
    )
    assert out["window"] == 3
    assert out["ma"].shape == (5,)


def test_moving_average_stats_short_series():
    # window_s/dt = 10 rounds to the nearest odd count (11); edge-padding keeps
    # length 2 and the smoothed peak sits between the two samples.
    out = mesh_field.moving_average_stats(np.array([2.0, 4.0]), dt=1.0, window_s=10.0)
    assert out["window"] == 11
    assert out["ma"].shape == (2,)
    assert 2.9 <= out["ma_max"] <= 3.1


def test_moving_average_stats_empty():
    out = mesh_field.moving_average_stats(np.array([]), dt=1.0, window_s=5.0)
    assert np.isnan(out["mean"]) and np.isnan(out["ma_max"])
