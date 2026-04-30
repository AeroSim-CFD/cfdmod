import pathlib

import numpy as np
import pytest
from lnas import LnasFormat

from cfdmod.use_cases.roughness_gen import RadialParams, radial_pattern
from cfdmod.use_cases.roughness_gen.parameters import ElementParams


def _make_flat_surface(tmp_path: pathlib.Path, size: float = 300.0, z: float = 0.0) -> pathlib.Path:
    triangles = np.array(
        [
            [[-size, -size, z], [size, -size, z], [size, size, z]],
            [[-size, -size, z], [size, size, z], [-size, size, z]],
        ],
        dtype=np.float32,
    )
    normals = np.array([[0, 0, 1], [0, 0, 1]], dtype=np.float32)
    path = tmp_path / "flat_surface.stl"
    LnasFormat.from_triangles(triangles=triangles, normals=normals).geometry.export_stl(path)
    return path


def test_radial_params_defaults():
    params = RadialParams(
        element_params=ElementParams(height=1.0, width=2.0),
        r_start=100.0,
        r_end=500.0,
        radial_spacing=10.0,
        arc_spacing=15.0,
        surfaces={"terrain": "fixtures/tests/loft/terrain.stl"},
    )
    assert params.ring_offset_distance == 0.0
    assert params.center == (0.0, 0.0)
    assert params.r_start == 100.0
    assert params.r_end == 500.0


def test_radial_pattern_output_shape(tmp_path):
    surface_path = _make_flat_surface(tmp_path, size=200.0, z=0.0)
    element_params = ElementParams(height=0.5, width=1.0)

    triangles, normals = radial_pattern(
        element_params=element_params,
        r_start=50.0,
        r_end=100.0,
        radial_spacing=20.0,
        arc_spacing=30.0,
        ring_offset_distance=5.0,
        center=(0.0, 0.0),
        surface_paths=[surface_path],
    )

    assert len(triangles) > 0
    assert len(triangles) == len(normals)
    assert len(triangles) % 2 == 0


def test_radial_pattern_normals_are_outward_radial(tmp_path):
    surface_path = _make_flat_surface(tmp_path, size=200.0, z=0.0)
    element_params = ElementParams(height=0.5, width=1.0)

    triangles, normals = radial_pattern(
        element_params=element_params,
        r_start=50.0,
        r_end=100.0,
        radial_spacing=20.0,
        arc_spacing=30.0,
        ring_offset_distance=0.0,
        center=(0.0, 0.0),
        surface_paths=[surface_path],
    )

    # Normals must lie in XY plane
    np.testing.assert_allclose(normals[:, 2], 0.0, atol=1e-6)

    # Normals must be unit vectors
    norms = np.linalg.norm(normals, axis=1)
    np.testing.assert_allclose(norms, 1.0, atol=1e-5)

    # Each fin centroid should be in the direction of its normal
    centroids = triangles.mean(axis=1)
    centroid_xy = centroids[:, :2]
    centroid_norms = np.linalg.norm(centroid_xy, axis=1, keepdims=True)
    centroid_dirs = centroid_xy / centroid_norms
    dot = (centroid_dirs * normals[:, :2]).sum(axis=1)
    assert np.all(dot > 0.0)


def test_radial_pattern_no_fins_outside_surface(tmp_path):
    surface_path = _make_flat_surface(tmp_path, size=30.0, z=0.0)
    element_params = ElementParams(height=0.5, width=1.0)

    # r_start > surface size -> no valid Z -> no fins
    triangles, normals = radial_pattern(
        element_params=element_params,
        r_start=100.0,
        r_end=150.0,
        radial_spacing=10.0,
        arc_spacing=20.0,
        ring_offset_distance=0.0,
        center=(0.0, 0.0),
        surface_paths=[surface_path],
    )

    assert len(triangles) == 0
    assert len(normals) == 0
