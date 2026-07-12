"""Tests for cfdmod.building.modes_report structural figures."""

from __future__ import annotations

import matplotlib

matplotlib.use("Agg")  # headless

import numpy as np  # noqa: E402
import pytest  # noqa: E402

from cfdmod.building import modes_report  # noqa: E402
from cfdmod.dynamics.structural import BuildingStructuralData  # noqa: E402

pytestmark = pytest.mark.unit


def _structure(n_floors: int = 4, n_modes: int = 3) -> BuildingStructuralData:
    z = np.linspace(0.0, 100.0, n_floors)
    return BuildingStructuralData(
        mode_shapes=np.random.default_rng(0).normal(size=(n_floors, n_modes, 3)),
        natural_frequencies=2 * np.pi * np.array([0.2, 0.5, 0.9])[:n_modes],
        floor_points=np.column_stack([np.zeros(n_floors), np.zeros(n_floors), z]),
        cm_positions=np.zeros((n_floors, 2)),
        floors_mass=np.linspace(5000.0, 3000.0, n_floors),
        floors_radius=np.full(n_floors, 8.0),
    )


def test_plot_mode_shape_returns_fig_ax():
    st = _structure()
    fig, ax = modes_report.plot_mode_shape(st, mode=0, rotation_scale=st.floors_radius)
    assert fig is not None and ax is not None
    # three lines: DX, DY, RZ
    assert len(ax.get_lines()) == 3


def test_plot_mode_shape_out_of_range():
    with pytest.raises(IndexError):
        modes_report.plot_mode_shape(_structure(n_modes=2), mode=5)


def test_plot_floor_mass():
    fig, ax = modes_report.plot_floor_mass(_structure(), use_height=True)
    assert fig is not None
    line = ax.get_lines()[0]
    assert len(line.get_xdata()) == 4


def test_plot_natural_frequencies_converts_to_hz():
    st = _structure(n_modes=3)
    fig, ax = modes_report.plot_natural_frequencies(st)
    ydata = ax.get_lines()[0].get_ydata()
    # wp = 2*pi*f stored -> Hz recovered
    assert np.allclose(np.sort(ydata), [0.2, 0.5, 0.9], atol=1e-9)
