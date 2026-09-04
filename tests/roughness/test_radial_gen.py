import pathlib

import numpy as np
import pytest
from lnas import LnasFormat

from cfdmod.roughness import RadialParams, radial_pattern
from cfdmod.roughness.parameters import ElementParams


def _make_flat_surface(
    tmp_path: pathlib.Path, size: float = 300.0, z: float = 0.0
) -> pathlib.Path:
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


def _flat_vertices(size: float = 200.0, z: float = 3.0) -> np.ndarray:
    return np.array(
        [[-size, -size, z], [size, -size, z], [size, size, z], [-size, size, z]],
        dtype=np.float64,
    )


# Geometry of a 7-fin pattern (rings at r = 50 and r = 70) over a flat surface at
# z = 3, pinned so the batched fin assembly cannot drift from the per-fin one.
EXPECTED_FINS = np.array(
    [
        [[50.0, -0.5, 3.0], [50.0, 0.5, 3.0], [50.0, 0.5, 3.5]],
        [[50.0, -0.5, 3.0], [50.0, 0.5, 3.5], [50.0, -0.5, 3.5]],
        [[-24.566988, 43.55127, 3.0], [-25.433012, 43.05127, 3.0], [-25.433012, 43.05127, 3.5]],
        [[-24.566988, 43.55127, 3.0], [-25.433012, 43.05127, 3.5], [-24.566988, 43.55127, 3.5]],
        [[-25.433012, -43.05127, 3.0], [-24.566988, -43.55127, 3.0], [-24.566988, -43.55127, 3.5]],
        [[-25.433012, -43.05127, 3.0], [-24.566988, -43.55127, 3.5], [-25.433012, -43.05127, 3.5]],
        [[69.857185, 4.4970245, 3.0], [69.78582, 5.4944744, 3.0], [69.78582, 5.4944744, 3.5]],
        [[69.857185, 4.4970245, 3.0], [69.78582, 5.4944744, 3.5], [69.857185, 4.4970245, 3.5]],
        [[-4.4970245, 69.857185, 3.0], [-5.4944744, 69.78582, 3.0], [-5.4944744, 69.78582, 3.5]],
        [[-4.4970245, 69.857185, 3.0], [-5.4944744, 69.78582, 3.5], [-4.4970245, 69.857185, 3.5]],
        [
            [-69.857185, -4.4970245, 3.0],
            [-69.78582, -5.4944744, 3.0],
            [-69.78582, -5.4944744, 3.5],
        ],
        [
            [-69.857185, -4.4970245, 3.0],
            [-69.78582, -5.4944744, 3.5],
            [-69.857185, -4.4970245, 3.5],
        ],
        [[4.4970245, -69.857185, 3.0], [5.4944744, -69.78582, 3.0], [5.4944744, -69.78582, 3.5]],
        [[4.4970245, -69.857185, 3.0], [5.4944744, -69.78582, 3.5], [4.4970245, -69.857185, 3.5]],
    ],
    dtype=np.float32,
)

EXPECTED_NORMALS = np.array(
    [
        [1.0, 0.0, 0.0],
        [1.0, 0.0, 0.0],
        [-0.5, 0.8660254, 0.0],
        [-0.5, 0.8660254, 0.0],
        [-0.5, -0.8660254, 0.0],
        [-0.5, -0.8660254, 0.0],
        [0.99745005, 0.07136784, 0.0],
        [0.99745005, 0.07136784, 0.0],
        [-0.07136784, 0.99745005, 0.0],
        [-0.07136784, 0.99745005, 0.0],
        [-0.99745005, -0.07136784, 0.0],
        [-0.99745005, -0.07136784, 0.0],
        [0.07136784, -0.99745005, 0.0],
        [0.07136784, -0.99745005, 0.0],
    ],
    dtype=np.float32,
)


def test_radial_pattern_geometry_is_pinned():
    triangles, normals = radial_pattern(
        element_params=ElementParams(height=0.5, width=1.0),
        r_start=50.0,
        r_end=70.0,
        radial_spacing=20.0,
        arc_spacing=100.0,
        ring_offset_distance=5.0,
        center=(0.0, 0.0),
        surfaces=[_flat_vertices()],
    )

    np.testing.assert_allclose(triangles, EXPECTED_FINS, atol=1e-5)
    np.testing.assert_allclose(normals, EXPECTED_NORMALS, atol=1e-6)


def test_radial_pattern_accepts_in_memory_geometry(tmp_path):
    surface_path = _make_flat_surface(tmp_path, size=200.0, z=0.0)
    lnas = LnasFormat.from_file(surface_path)
    kwargs = dict(
        element_params=ElementParams(height=0.5, width=1.0),
        r_start=50.0,
        r_end=100.0,
        radial_spacing=20.0,
        arc_spacing=30.0,
        ring_offset_distance=5.0,
        center=(0.0, 0.0),
    )

    from_path = radial_pattern(surfaces=[surface_path], **kwargs)
    from_format = radial_pattern(surfaces=[lnas], **kwargs)
    from_geometry = radial_pattern(surfaces=[lnas.geometry], **kwargs)
    from_array = radial_pattern(surfaces=[lnas.geometry.vertices], **kwargs)

    for other in (from_format, from_geometry, from_array):
        np.testing.assert_allclose(other[0], from_path[0], atol=1e-5)
        np.testing.assert_allclose(other[1], from_path[1], atol=1e-6)


def test_radial_pattern_keeps_the_legacy_surface_paths_keyword(tmp_path):
    surface_path = _make_flat_surface(tmp_path, size=200.0, z=0.0)
    kwargs = dict(
        element_params=ElementParams(height=0.5, width=1.0),
        r_start=50.0,
        r_end=100.0,
        radial_spacing=20.0,
        arc_spacing=30.0,
        ring_offset_distance=0.0,
        center=(0.0, 0.0),
    )

    legacy, _ = radial_pattern(surface_paths=[surface_path], **kwargs)
    current, _ = radial_pattern(surfaces=[surface_path], **kwargs)
    np.testing.assert_array_equal(legacy, current)

    with pytest.raises(ValueError, match="not both"):
        radial_pattern(surfaces=[surface_path], surface_paths=[surface_path], **kwargs)
    with pytest.raises(ValueError, match="`surfaces` is required"):
        radial_pattern(**kwargs)


def test_radial_pattern_with_no_rings_is_empty(tmp_path):
    triangles, normals = radial_pattern(
        element_params=ElementParams(height=0.5, width=1.0),
        r_start=200.0,
        r_end=100.0,
        radial_spacing=20.0,
        arc_spacing=30.0,
        ring_offset_distance=0.0,
        center=(0.0, 0.0),
        surfaces=[_flat_vertices()],
    )

    assert len(triangles) == 0
    assert len(normals) == 0
