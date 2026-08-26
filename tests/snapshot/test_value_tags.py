"""Tests for the ValueTagsConfig positioning modes and point transformation."""

from __future__ import annotations

import numpy as np
import pytest

# The snapshot package imports pyvista + IPython at load; skip where absent.
snapshot = pytest.importorskip("cfdmod.snapshot")

pytestmark = pytest.mark.unit

ValueTagsConfig = snapshot.ValueTagsConfig
TransformationConfig = snapshot.TransformationConfig
transform_points = snapshot.transform_points


def test_points_mode_clears_grid_and_floating_options():
    cfg = ValueTagsConfig(
        points=[(0.0, 0.0, 1.0), (2.0, 3.0, 4.0)],
        x=[1.0],
        y=[2.0],
        spacing=25,
        padding=(5, 10),
    )
    assert cfg.points == [(0.0, 0.0, 1.0), (2.0, 3.0, 4.0)]
    assert cfg.x is None
    assert cfg.y is None
    assert cfg.spacing is None
    assert cfg.padding is None


def test_exact_position_without_both_axes_raises():
    with pytest.raises(ValueError, match="Both must be specified"):
        ValueTagsConfig(x=[1.0])
    with pytest.raises(ValueError, match="Both must be specified"):
        ValueTagsConfig(y=[1.0])


def test_exact_position_takes_precedence_over_floating():
    cfg = ValueTagsConfig(x=[1.0], y=[2.0], spacing=25, padding=(5, 10))
    assert cfg.x == [1.0]
    assert cfg.y == [2.0]
    assert cfg.spacing is None
    assert cfg.padding is None


def test_floating_position_without_both_options_raises():
    with pytest.raises(ValueError, match="Both must be specified"):
        ValueTagsConfig(spacing=25)
    with pytest.raises(ValueError, match="Both must be specified"):
        ValueTagsConfig(padding=(5, 10))


def test_transform_points_translation_only():
    transformation = TransformationConfig(translate=(1.0, 2.0, 3.0))
    out = transform_points(
        np.array([[0.0, 0.0, 0.0], [1.0, 1.0, 1.0]]), transformation, center=(0.0, 0.0, 0.0)
    )
    np.testing.assert_allclose(out, [[1.0, 2.0, 3.0], [2.0, 3.0, 4.0]], atol=1e-9)


def test_transform_points_rotates_about_given_center():
    transformation = TransformationConfig(rotate=(0.0, 0.0, 90.0))
    out = transform_points(np.array([[1.0, 0.0, 0.0]]), transformation, center=(0.0, 0.0, 0.0))
    np.testing.assert_allclose(out, [[0.0, 1.0, 0.0]], atol=1e-6)

    # rotating about the point itself leaves it untouched
    out = transform_points(np.array([[1.0, 0.0, 0.0]]), transformation, center=(1.0, 0.0, 0.0))
    np.testing.assert_allclose(out, [[1.0, 0.0, 0.0]], atol=1e-6)


def test_transform_points_matches_transform_mesh_on_same_coordinates():
    pv = pytest.importorskip("pyvista")
    transformation = TransformationConfig(rotate=(10.0, 20.0, 30.0), translate=(1.0, -2.0, 3.0))
    raw = np.array([[0.0, 0.0, 0.0], [1.0, 2.0, 3.0], [-4.0, 5.0, 6.0]])

    mesh = pv.PolyData(raw.copy())
    center = mesh.center
    snapshot.transform_mesh(mesh, transformation)

    out = transform_points(raw.copy(), transformation, center=center)
    np.testing.assert_allclose(out, mesh.points, atol=1e-6)
