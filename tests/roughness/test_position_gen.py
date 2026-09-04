import pathlib

import numpy as np
import pytest

from cfdmod.io.geometry.STL import read_stl
from cfdmod.roughness import (
    BoundingBox,
    ElementParams,
    PositionParams,
    SpacingParams,
    position_pattern,
)
from cfdmod.roughness.run import run_position

TRIANGLES_PER_ELEMENT = 2

UNBOUNDED_BOX = BoundingBox(
    start=(float("-inf"), float("-inf"), float("-inf")),
    end=(float("inf"), float("inf"), float("inf")),
)


def _plane(size: float = 100.0, z0: float = 0.0, slope_x: float = 0.0, slope_y: float = 0.0):
    """Vertices of a plane z = z0 + slope_x * x + slope_y * y over [-size, size]^2."""
    corners = np.array([[-size, -size], [size, -size], [size, size], [-size, size]])
    z = z0 + slope_x * corners[:, 0] + slope_y * corners[:, 1]
    return np.column_stack([corners, z])


def _element_bases(triangles: np.ndarray) -> np.ndarray:
    """Lowest Z of each element."""
    return triangles.reshape(-1, TRIANGLES_PER_ELEMENT * 3, 3)[:, :, 2].min(axis=1)


def _spacing(offset_direction: str = "x", line_offset: float = 0.0) -> SpacingParams:
    return SpacingParams(
        spacing=(20.0, 20.0), line_offset=line_offset, offset_direction=offset_direction
    )


def test_flat_surface_puts_every_element_base_on_it():
    triangles, normals = position_pattern(
        element_params=ElementParams(height=2.0, width=6.0),
        spacing_params=_spacing(),
        bounding_box=UNBOUNDED_BOX,
        surfaces=[_plane(size=100.0, z0=7.5)],
    )

    assert len(triangles) > 0
    assert len(triangles) == len(normals)
    assert len(triangles) % TRIANGLES_PER_ELEMENT == 0
    np.testing.assert_allclose(_element_bases(triangles), 7.5, atol=1e-4)
    # The element keeps its height above the surface it was draped onto.
    heights = triangles.reshape(-1, TRIANGLES_PER_ELEMENT * 3, 3)[:, :, 2]
    np.testing.assert_allclose(heights.max(axis=1) - heights.min(axis=1), 2.0, atol=1e-4)


def test_elements_follow_a_slope_along_x():
    slope = 0.05
    triangles, _ = position_pattern(
        element_params=ElementParams(height=1.0, width=4.0),
        spacing_params=_spacing(),
        bounding_box=UNBOUNDED_BOX,
        surfaces=[_plane(size=100.0, z0=10.0, slope_x=slope)],
    )

    per_element = triangles.reshape(-1, TRIANGLES_PER_ELEMENT * 3, 3)
    x = per_element[:, :, 0].min(axis=1)
    np.testing.assert_allclose(_element_bases(triangles), 10.0 + slope * x, atol=1e-3)


def test_element_is_seated_on_the_low_side_of_a_cross_slope():
    """An element spans its width in Y, so a Y slope must not leave it floating."""
    slope = 0.05
    width = 8.0
    triangles, _ = position_pattern(
        element_params=ElementParams(height=1.0, width=width),
        spacing_params=_spacing(),
        bounding_box=UNBOUNDED_BOX,
        surfaces=[_plane(size=100.0, z0=10.0, slope_y=slope)],
    )

    per_element = triangles.reshape(-1, TRIANGLES_PER_ELEMENT * 3, 3)
    y_low = per_element[:, :, 1].min(axis=1)
    # The lift is the surface Z at the low edge, not at the element centre.
    np.testing.assert_allclose(_element_bases(triangles), 10.0 + slope * y_low, atol=1e-3)


def test_bounding_box_clips_the_array():
    surfaces = [_plane(size=100.0, z0=0.0)]
    element_params = ElementParams(height=1.0, width=4.0)

    full, _ = position_pattern(
        element_params=element_params,
        spacing_params=_spacing(),
        bounding_box=UNBOUNDED_BOX,
        surfaces=surfaces,
    )
    clipped, _ = position_pattern(
        element_params=element_params,
        spacing_params=_spacing(),
        bounding_box=BoundingBox(start=(-20.0, -20.0, 0.0), end=(20.0, 20.0, 0.0)),
        surfaces=surfaces,
    )

    assert 0 < len(clipped) < len(full)
    assert clipped[:, :, 0].min() >= -20.0 - 1e-6
    assert clipped[:, :, 0].max() <= 20.0 + 1e-6
    assert clipped[:, :, 1].min() >= -20.0 - 1e-6
    assert clipped[:, :, 1].max() <= 20.0 + 1e-6


def test_array_stays_inside_the_surface_extent():
    size = 100.0
    triangles, _ = position_pattern(
        element_params=ElementParams(height=1.0, width=4.0),
        spacing_params=_spacing(offset_direction="y", line_offset=7.0),
        bounding_box=UNBOUNDED_BOX,
        surfaces=[_plane(size=size, z0=0.0)],
    )

    assert triangles[:, :, 0].min() >= -size - 1e-6
    assert triangles[:, :, 0].max() <= size + 1e-6
    assert triangles[:, :, 1].min() >= -size - 1e-6
    assert triangles[:, :, 1].max() <= size + 1e-6


def test_elements_outside_every_surface_are_dropped():
    """Two surfaces with a hole between them: nothing is left sitting at z = 0."""
    left = _plane(size=40.0, z0=5.0)
    right = left.copy()
    right[:, 0] += 400.0
    right[:, 2] = 9.0

    triangles, normals = position_pattern(
        element_params=ElementParams(height=1.0, width=4.0),
        spacing_params=_spacing(),
        bounding_box=UNBOUNDED_BOX,
        surfaces=[left, right],
        max_points=None,
    )

    assert len(triangles) == len(normals)
    bases = _element_bases(triangles)
    # The pooled triangulation bridges the gap linearly, so every base sits
    # between the two plane heights. None is left at the untouched z = 0.
    assert bases.min() >= 5.0 - 1e-3
    assert bases.max() <= 9.0 + 1e-3


def test_box_outside_the_surfaces_is_rejected():
    with pytest.raises(ValueError, match="too small for a single element"):
        position_pattern(
            element_params=ElementParams(height=1.0, width=4.0),
            spacing_params=_spacing(),
            bounding_box=BoundingBox(start=(500.0, 500.0, 0.0), end=(600.0, 600.0, 0.0)),
            surfaces=[_plane(size=100.0)],
        )


def test_footprint_samples_must_span_the_element():
    with pytest.raises(ValueError, match="footprint_samples must be at least 2"):
        position_pattern(
            element_params=ElementParams(height=1.0, width=4.0),
            spacing_params=_spacing(),
            bounding_box=UNBOUNDED_BOX,
            surfaces=[_plane(size=100.0)],
            footprint_samples=1,
        )


def test_thinning_does_not_move_the_elements():
    rng = np.random.default_rng(7)
    grid = np.linspace(-400.0, 400.0, 220)
    xx, yy = np.meshgrid(grid, grid)
    xx = xx + rng.normal(0.0, 1.0, xx.shape)
    yy = yy + rng.normal(0.0, 1.0, yy.shape)
    zz = 750.0 + 15.0 * np.sin(xx / 180.0) + 9.0 * np.cos(yy / 260.0)
    terrain = np.stack([xx.ravel(), yy.ravel(), zz.ravel()], axis=1)

    kwargs = dict(
        element_params=ElementParams(height=2.0, width=6.0),
        spacing_params=_spacing(),
        bounding_box=UNBOUNDED_BOX,
        surfaces=[terrain],
    )
    full, _ = position_pattern(max_points=None, **kwargs)
    thinned, _ = position_pattern(max_points=5_000, **kwargs)

    assert full.shape == thinned.shape
    np.testing.assert_allclose(full[:, :, :2], thinned[:, :, :2], atol=1e-4)
    assert np.abs(_element_bases(full) - _element_bases(thinned)).max() < 0.2


def test_position_params_fixture_drives_the_routine():
    cfg = PositionParams.from_file(
        pathlib.Path("./fixtures/tests/roughness_gen/position_params.yaml")
    )
    triangles, normals = position_pattern(
        element_params=cfg.element_params,
        spacing_params=cfg.spacing_params,
        bounding_box=cfg.bounding_box,
        surfaces=[pathlib.Path(p) for p in cfg.surfaces.values()],
    )

    assert len(triangles) > 0
    assert len(triangles) == len(normals)
    # The fixture surfaces sit around z = 750 to 855; nothing may be left at 0.
    bases = _element_bases(triangles)
    assert bases.min() > 700.0
    assert bases.max() < 900.0


def test_run_position_writes_the_stl(tmp_path):
    cfg = PositionParams.from_file(
        pathlib.Path("./fixtures/tests/roughness_gen/position_params.yaml")
    )
    run_position(cfg, tmp_path)

    output = tmp_path / "positioned_elements.stl"
    assert output.exists()
    triangles, _ = read_stl(output)
    assert len(triangles) > 0
