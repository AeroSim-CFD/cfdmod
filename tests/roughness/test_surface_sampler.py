import pathlib

import numpy as np
import pytest
from lnas import LnasFormat, LnasGeometry

from cfdmod.roughness import SurfaceSampler, build_surface_sampler, load_surface_vertices


def _plane_triangles(size: float = 100.0, slope: float = 0.0, z0: float = 0.0) -> np.ndarray:
    """Two triangles covering [-size, size]^2 on the plane z = z0 + slope * x."""

    def z(x: float) -> float:
        return z0 + slope * x

    return np.array(
        [
            [[-size, -size, z(-size)], [size, -size, z(size)], [size, size, z(size)]],
            [[-size, -size, z(-size)], [size, size, z(size)], [-size, size, z(-size)]],
        ],
        dtype=np.float32,
    )


def _lnas_from_triangles(triangles: np.ndarray) -> LnasFormat:
    normals = np.tile(np.array([0.0, 0.0, 1.0], dtype=np.float32), (len(triangles), 1))
    return LnasFormat.from_triangles(triangles=triangles, normals=normals)


def _synthetic_terrain(side: int, seed: int = 0) -> np.ndarray:
    """A jittered grid terrain, dense enough to trigger thinning."""
    rng = np.random.default_rng(seed)
    grid = np.linspace(-500.0, 500.0, side)
    xx, yy = np.meshgrid(grid, grid)
    xx = xx + rng.normal(0.0, 1.0, xx.shape)
    yy = yy + rng.normal(0.0, 1.0, yy.shape)
    zz = 800.0 + 20.0 * np.sin(xx / 150.0) + 12.0 * np.cos(yy / 220.0)
    return np.stack([xx.ravel(), yy.ravel(), zz.ravel()], axis=1)


def test_sampler_accepts_every_surface_input_kind(tmp_path):
    triangles = _plane_triangles(size=50.0, slope=0.1, z0=3.0)
    lnas = _lnas_from_triangles(triangles)

    stl_path = tmp_path / "plane.stl"
    lnas.geometry.export_stl(stl_path)
    lnas_path = tmp_path / "plane.lnas"
    lnas.to_file(lnas_path)

    query = np.array([[0.0, 0.0], [10.0, -20.0], [-30.0, 5.0]])
    inputs = [
        stl_path,
        str(stl_path),
        lnas_path,
        lnas,
        lnas.geometry,
        LnasGeometry(vertices=lnas.geometry.vertices, triangles=lnas.geometry.triangles),
        lnas.geometry.vertices,
    ]

    results = [build_surface_sampler([s]).sample(query) for s in inputs]
    for result in results[1:]:
        np.testing.assert_allclose(result, results[0], atol=1e-6)

    # The plane is z = 3 + 0.1 x, so the sample is exact up to float32 vertices.
    np.testing.assert_allclose(results[0], 3.0 + 0.1 * query[:, 0], atol=1e-4)


def test_sampler_is_nan_outside_the_surface():
    sampler = build_surface_sampler([_plane_triangles(size=10.0).reshape(-1, 3)])
    z = sampler.sample(np.array([[0.0, 0.0], [100.0, 0.0]]))

    assert not np.isnan(z[0])
    assert np.isnan(z[1])


def test_sampler_pools_multiple_surfaces():
    left = _plane_triangles(size=10.0, z0=1.0).reshape(-1, 3)
    right = left.copy()
    right[:, 0] += 40.0
    right[:, 2] = 2.0

    sampler = build_surface_sampler([left, right])
    z = sampler.sample(np.array([[-9.0, 0.0], [39.0, 0.0]]))

    np.testing.assert_allclose(z, [1.0, 2.0], atol=1e-4)


def test_thinning_respects_the_cap_and_keeps_the_domain():
    terrain = _synthetic_terrain(side=200)
    max_points = 5_000

    unbounded = build_surface_sampler([terrain], max_points=None)
    bounded = build_surface_sampler([terrain], max_points=max_points)

    assert unbounded.n_points > max_points
    assert bounded.n_points <= max_points

    grid = np.linspace(-499.0, 499.0, 90)
    xx, yy = np.meshgrid(grid, grid)
    query = np.stack([xx.ravel(), yy.ravel()], axis=1)

    z_unbounded = unbounded.sample(query)
    z_bounded = bounded.sample(query)

    # Thinning keeps the surface boundary, so no position that had a value loses it.
    assert not np.any(~np.isnan(z_unbounded) & np.isnan(z_bounded))

    both = ~np.isnan(z_unbounded) & ~np.isnan(z_bounded)
    assert both.sum() > 0
    # The surface varies by tens of metres; the thinned drape stays within
    # centimetres of the full one.
    assert np.abs(z_unbounded[both] - z_bounded[both]).max() < 0.2


def test_thinning_keeps_the_edge_as_accurate_as_the_middle():
    """The rim of the surface is kept at full density on purpose.

    Thinning only the interior leaves the triangulation spanning the gap between
    a sparse boundary and the thinned interior with very wide triangles, which
    puts the interpolated Z metres off exactly where the roughness array meets
    the edge of the terrain.
    """
    terrain = _synthetic_terrain(side=200)
    bounded = build_surface_sampler([terrain], max_points=5_000)

    def truth(xy: np.ndarray) -> np.ndarray:
        return 800.0 + 20.0 * np.sin(xy[:, 0] / 150.0) + 12.0 * np.cos(xy[:, 1] / 220.0)

    grid = np.linspace(-499.0, 499.0, 90)
    xx, yy = np.meshgrid(grid, grid)
    query = np.stack([xx.ravel(), yy.ravel()], axis=1)
    error = np.abs(bounded.sample(query) - truth(query))

    radius = np.abs(query).max(axis=1)
    edge = radius > 490.0
    middle = radius < 400.0

    assert np.nanmax(error[edge]) < 2.0 * np.nanmax(error[middle])


def test_thinning_is_deterministic():
    terrain = _synthetic_terrain(side=120)
    first = build_surface_sampler([terrain], max_points=2_000)
    second = build_surface_sampler([terrain], max_points=2_000)

    np.testing.assert_array_equal(first.points, second.points)


def test_no_thinning_below_the_cap():
    terrain = _synthetic_terrain(side=40)
    sampler = build_surface_sampler([terrain], max_points=None)
    capped = build_surface_sampler([terrain], max_points=10_000)

    assert sampler.n_points == len(np.unique(terrain, axis=0))
    np.testing.assert_array_equal(sampler.points, capped.points)


def test_sampler_rejects_bad_inputs():
    plane = _plane_triangles(size=10.0).reshape(-1, 3)

    with pytest.raises(ValueError, match="At least one surface"):
        build_surface_sampler([])
    with pytest.raises(ValueError, match="max_points must be at least 3"):
        build_surface_sampler([plane], max_points=2)
    with pytest.raises(ValueError, match=r"shape \(N, 3\)"):
        build_surface_sampler([np.zeros((4, 2))])
    with pytest.raises(TypeError, match="Unsupported surface input"):
        build_surface_sampler([42])
    with pytest.raises(ValueError, match="at least 3 surface points"):
        SurfaceSampler(np.zeros((2, 3)))
    with pytest.raises(ValueError, match=r"shape \(N, 2\)"):
        build_surface_sampler([plane]).sample(np.zeros((4, 3)))


def test_points_are_handed_out_read_only():
    """A mutated points array would silently disagree with the interpolator."""
    plane = _plane_triangles(size=10.0).reshape(-1, 3).astype(np.float64)
    sampler = build_surface_sampler([plane])

    with pytest.raises(ValueError):
        sampler.points[0, 2] = 99.0
    # The caller's own array is left alone.
    caller_owned = np.unique(plane, axis=0)
    SurfaceSampler(caller_owned)
    caller_owned[0, 2] = 99.0


def test_sample_of_no_positions_is_empty():
    sampler = build_surface_sampler([_plane_triangles(size=10.0).reshape(-1, 3)])
    assert sampler.sample(np.empty((0, 2))).shape == (0,)


def test_load_surface_vertices_reads_each_surface_once(tmp_path):
    lnas = _lnas_from_triangles(_plane_triangles(size=10.0, z0=4.0))
    path = tmp_path / "plane.lnas"
    lnas.to_file(path)

    vertices = load_surface_vertices([path, lnas.geometry.vertices])

    assert len(vertices) == 2
    assert all(v.shape[1] == 3 and v.dtype == np.float64 for v in vertices)


def test_sampler_matches_the_repository_fixture_surfaces():
    surfaces = [
        pathlib.Path("./fixtures/tests/roughness_gen/disk/disk.lnas"),
        pathlib.Path("./fixtures/tests/roughness_gen/loft/loft.lnas"),
    ]
    sampler = build_surface_sampler(surfaces)
    z = sampler.sample(np.array([[0.0, 0.0], [1_000_000.0, 0.0]]))

    assert 700.0 < z[0] < 900.0
    assert np.isnan(z[1])
