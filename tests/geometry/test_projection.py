"""Unit tests for the projected (shadow) area of a surface along a wind direction."""

from __future__ import annotations

import numpy as np
import pytest

pytest.importorskip("shapely")

from cfdmod.geometry.projection import projected_area  # noqa: E402

pytestmark = pytest.mark.unit

#: Vertices snap to a grid of 1e-6 x the model extent: lengths and areas are
#: good to about 1e-6 relative, so the checks allow 1e-5.
REL = 1e-5

_QUADS = ((0, 1, 3, 2), (4, 5, 7, 6), (0, 1, 5, 4), (2, 3, 7, 6), (0, 2, 6, 4), (1, 3, 7, 5))


def _box(x0, x1, y0, y1, z0, z1) -> np.ndarray:
    """Closed box as a ``(12, 3, 3)`` triangle array (corner index = 4 iz + 2 iy + ix)."""
    corners = np.array([[x, y, z] for z in (z0, z1) for y in (y0, y1) for x in (x0, x1)], float)
    tris = [(a, b, c) for a, b, c, d in _QUADS] + [(a, c, d) for a, b, c, d in _QUADS]
    return corners[np.array(tris)]


def _prism(n_sides: int, radius: float, height: float, phase_deg: float) -> np.ndarray:
    """Side walls of a regular ``n_sides`` prism (no caps) as a triangle array."""
    th = np.deg2rad(phase_deg) + 2 * np.pi * np.arange(n_sides) / n_sides
    ring = np.stack([radius * np.cos(th), radius * np.sin(th)], axis=1)
    tris = []
    for i in range(n_sides):
        (xa, ya), (xb, yb) = ring[i], ring[(i + 1) % n_sides]
        a0, b0, a1, b1 = (xa, ya, 0), (xb, yb, 0), (xa, ya, height), (xb, yb, height)
        tris += [(a0, b0, b1), (a0, b1, a1)]
    return np.array(tris, float)


@pytest.mark.parametrize("direction", [0.0, 30.0, 90.0, 135.0, 200.0, 315.0])
def test_box_area_is_the_frontal_rectangle(direction):
    """1 (x) x 2 (y) x 3 (z) box: width 2 |cos| + 1 |sin|, front and back counted once."""
    r = projected_area(_box(0, 1, 0, 2, 0, 3), direction)
    a = np.deg2rad(direction)
    width = 2 * abs(np.cos(a)) + abs(np.sin(a))
    assert r.width == pytest.approx(width, rel=REL)
    assert r.area == pytest.approx(3 * width, rel=REL)


@pytest.mark.parametrize("direction", [0.0, 7.0, 45.0, 100.0])
def test_convex_prism_area_is_its_vertex_span_times_height(direction):
    tris = _prism(16, radius=10.0, height=50.0, phase_deg=3.0)
    r = projected_area(tris, direction)
    u = tris.reshape(-1, 3) @ r.u_axis
    assert r.area == pytest.approx((u.max() - u.min()) * 50.0, rel=REL)


def test_surface_hidden_behind_another_counts_once():
    """A leaf wall in front of a tower adds only the part sticking out of it."""
    tower = _box(0, 10, 0, 20, 0, 30)
    leaf = _box(-3, -2.9, 15, 25, 0, 30)  # 5 m of its 10 m beyond the tower side
    r = projected_area(np.concatenate([tower, leaf]), 0.0)
    assert r.area == pytest.approx(25 * 30, rel=REL)


def test_opening_is_an_interior_ring_and_is_not_counted():
    """A frame (4 x 5 outer, 2 x 3 opening) seen through its opening."""
    frame = np.concatenate(
        [
            _box(0, 1, 0, 1, 0, 5),  # left column
            _box(0, 1, 3, 4, 0, 5),  # right column
            _box(0, 1, 1, 3, 0, 1),  # bottom beam
            _box(0, 1, 1, 3, 4, 5),  # top beam
        ]
    )
    r = projected_area(frame, 0.0)
    assert r.area == pytest.approx(4 * 5 - 2 * 3, rel=REL)
    assert r.shadow.geom_type == "Polygon"
    assert len(r.shadow.interiors) == 1


def test_separate_bodies_give_a_multipolygon_with_the_summed_area():
    r = projected_area(np.concatenate([_box(0, 1, 0, 2, 0, 3), _box(0, 1, 5, 6, 0, 2)]), 0.0)
    assert r.shadow.geom_type == "MultiPolygon"
    assert r.area == pytest.approx(2 * 3 + 1 * 2, rel=REL)


def test_band_areas_clip_the_shadow_by_z():
    box = _box(0, 1, 0, 2, 0, 3)
    r = projected_area(box, 0.0, z_intervals=[0, 1, 2, 3])
    np.testing.assert_allclose(r.band_areas, [2, 2, 2], rtol=REL)
    np.testing.assert_array_equal(r.z_intervals, [0, 1, 2, 3])

    r = projected_area(box, 0.0, z_intervals=[-1, 0, 1.5, 10])
    np.testing.assert_allclose(r.band_areas, [0, 3, 3], rtol=REL, atol=1e-9)
    assert r.band_areas.sum() == pytest.approx(r.area, rel=REL)


def test_u_axis_is_the_right_hand_of_an_upstream_observer():
    box = _box(0, 1, 2, 4, 0, 3)
    r = projected_area(box, 0.0)  # wind towards +x: right hand of the observer is -y
    np.testing.assert_allclose(r.u_axis, [0, -1, 0], atol=1e-12)
    u0, z0, u1, z1 = r.shadow.bounds
    assert (u0, z0, u1, z1) == pytest.approx((-4, 0, -2, 3), abs=r.grid_size)

    r = projected_area(box, 90.0)  # wind towards +y: right hand is +x
    np.testing.assert_allclose(r.u_axis, [1, 0, 0], atol=1e-12)


def test_grid_size_defaults_to_the_relative_grid():
    r = projected_area(_box(0, 1, 0, 200, 0, 50), 0.0)
    assert r.grid_size == pytest.approx(200e-6)
    assert projected_area(_box(0, 1, 0, 2, 0, 3), 0.0, grid_size=1e-3).grid_size == 1e-3


@pytest.mark.parametrize(
    ("kwargs", "match"),
    [
        ({"z_intervals": [0, 2, 1]}, "z_intervals"),
        ({"z_intervals": [1]}, "z_intervals"),
        ({"z_intervals": [[0, 1], [1, 2]]}, "z_intervals"),
        ({"grid_size": 0.0}, "grid_size"),
    ],
)
def test_bad_arguments_raise(kwargs, match):
    with pytest.raises(ValueError, match=match):
        projected_area(_box(0, 1, 0, 2, 0, 3), 0.0, **kwargs)


def test_empty_geometry_raises():
    with pytest.raises(ValueError, match="no triangles"):
        projected_area(np.zeros((0, 3, 3)), 0.0)
