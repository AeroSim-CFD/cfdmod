"""Tests for the nodal -> per-floor aggregation core."""

from __future__ import annotations

import numpy as np
import pytest

from cfdmod.dynamics.imports import NodalModel, aggregate_to_building
from cfdmod.dynamics.imports._textnum import to_float


def _model(*, with_massless: bool = False) -> NodalModel:
    # Two slabs (z=0, z=3), each two nodes on the x-axis with unequal DX so the
    # mass-weighted mean is a known value.
    coords = [[0.0, 0.0, 0.0], [2.0, 0.0, 0.0], [0.0, 0.0, 3.0], [2.0, 0.0, 3.0]]
    mass = [1.0, 1.0, 1.0, 1.0]
    # mode 0: node DX = [0, 2] per slab -> mass-weighted floor DX = 1.0
    shapes = [
        [[0.0, 0.0, 0.0]],
        [[2.0, 0.0, 0.0]],
        [[0.0, 0.0, 0.0]],
        [[2.0, 0.0, 0.0]],
    ]
    if with_massless:
        coords.append([5.0, 0.0, 6.0])
        mass.append(0.0)
        shapes.append([[9.0, 9.0, 9.0]])
    return NodalModel(
        coords=np.array(coords),
        mass=np.array(mass),
        periods=np.array([1.0]),
        shapes=np.array(shapes),
    )


def test_to_float_handles_comma_decimal_scientific():
    assert to_float("7,036E+00") == pytest.approx(7.036)
    assert to_float(" -1,25 ") == pytest.approx(-1.25)


def test_aggregation_geometry_and_shape():
    sd = aggregate_to_building(_model())
    assert sd.n_floors == 2
    assert sd.n_modes == 1
    # CoM at (1, 0); polar inertia 2 -> radius 1.
    np.testing.assert_allclose(np.asarray(sd.cm_positions)[0], [1.0, 0.0])
    np.testing.assert_allclose(sd.floors_radius, [1.0, 1.0])
    np.testing.assert_allclose(np.asarray(sd.floor_points)[:, 2], [0.0, 3.0])
    # Frequency 1 Hz -> wp = 2*pi.
    np.testing.assert_allclose(sd.natural_frequencies, [2 * np.pi])
    # Unit generalized mass after normalization.
    phi = np.asarray(sd.mode_shapes)
    m = np.asarray(sd.floors_mass)[:, None]
    r = np.asarray(sd.floors_radius)[:, None]
    m_gen = (m * (phi[:, :, 0] ** 2 + phi[:, :, 1] ** 2 + (r * phi[:, :, 2]) ** 2)).sum(axis=0)
    np.testing.assert_allclose(m_gen, [1.0], rtol=1e-12)


def test_massless_floor_is_dropped():
    sd = aggregate_to_building(_model(with_massless=True))
    assert sd.n_floors == 2  # the z=6 all-massless level is dropped


def test_massless_floor_raises_when_not_dropped():
    with pytest.raises(ValueError, match="zero total mass"):
        aggregate_to_building(_model(with_massless=True), drop_massless=False)


def test_floor_levels_collapse_intermediate_nodes():
    # Two real slabs at z=0 and z=3, plus an intermediate beam node at z=1.4
    # (nearer the z=0 slab). Given the authoritative levels, the beam node is
    # absorbed into the lower floor -> exactly two floors, not three.
    coords = np.array(
        [[0.0, 0.0, 0.0], [2.0, 0.0, 0.0], [1.0, 0.0, 1.4], [0.0, 0.0, 3.0], [2.0, 0.0, 3.0]]
    )
    model = NodalModel(
        coords=coords,
        mass=np.array([1.0, 1.0, 0.5, 1.0, 1.0]),
        periods=np.array([1.0]),
        shapes=np.zeros((5, 1, 3)),
    )
    sd = aggregate_to_building(model, floor_levels=[0.0, 3.0])
    assert sd.n_floors == 2
    np.testing.assert_allclose(np.asarray(sd.floor_points)[:, 2], [0.0, 3.0])
    # Lower floor carries its two slab nodes plus the beam node: 1 + 1 + 0.5.
    np.testing.assert_allclose(sorted(sd.floors_mass), [2.0, 2.5])
