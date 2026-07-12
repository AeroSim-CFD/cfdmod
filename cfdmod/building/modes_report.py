"""Figures for the structural model that feeds the dynamic-response stage.

Visualises a :class:`cfdmod.dynamics.structural.BuildingStructuralData`: mode
shapes (DX / DY / RZ per floor), the per-floor mass distribution, and the
natural-frequency spectrum. Presentation only -- the numbers come straight off
the structural object.
"""

from __future__ import annotations

import numpy as np

from cfdmod.plot_config import new_axes

# mode_shapes[:, mode, k] component order.
_COMPONENTS = ("DX", "DY", "RZ")


def _floor_axis(structure, use_height: bool):
    if use_height:
        return np.asarray(structure.floor_points, dtype=np.float64)[:, 2], "z [m]"
    return np.arange(structure.n_floors, dtype=np.float64), "floor"


def plot_mode_shape(
    structure,
    mode: int,
    *,
    use_height: bool = True,
    rotation_scale: float | np.ndarray | None = None,
):
    """Plot the DX / DY / RZ components of one mode vs floor (or height).

    ``rotation_scale`` multiplies the RZ (rotation) component so it plots on the
    same scale as the translations -- pass the floors' radius of gyration
    (``structure.floors_radius``) for a physically comparable amplitude.
    """
    shapes = np.asarray(structure.mode_shapes, dtype=np.float64)  # (n_floors, n_modes, 3)
    if not 0 <= mode < shapes.shape[1]:
        raise IndexError(f"mode {mode} out of range (n_modes={shapes.shape[1]})")
    y, ylabel = _floor_axis(structure, use_height)
    dx, dy, rz = shapes[:, mode, 0], shapes[:, mode, 1], shapes[:, mode, 2]
    if rotation_scale is not None:
        rz = rz * np.asarray(rotation_scale, dtype=np.float64)

    fig, ax = new_axes(xlabel="modal amplitude [-]", ylabel=ylabel, title=f"Mode {mode}")
    ax.plot(dx, y, "-o", ms=3, label="DX")
    ax.plot(dy, y, "--s", ms=3, label="DY")
    ax.plot(rz, y, ":^", ms=3, label="RZ" + ("*R" if rotation_scale is not None else ""))
    ax.legend(loc="best", frameon=False)
    return fig, ax


def plot_floor_mass(structure, *, use_height: bool = True):
    """Per-floor mass vs floor (or height)."""
    y, ylabel = _floor_axis(structure, use_height)
    mass = np.asarray(structure.floors_mass, dtype=np.float64)
    fig, ax = new_axes(xlabel="floor mass [kg]", ylabel=ylabel, title="Floor mass distribution")
    ax.plot(mass, y, "-o", ms=3)
    return fig, ax


def plot_natural_frequencies(structure):
    """Natural frequency (Hz) per mode number.

    ``natural_frequencies`` is stored as angular ``wp = 2*pi*f``; converted to Hz.
    """
    wp = np.asarray(structure.natural_frequencies, dtype=np.float64)
    f_hz = wp / (2.0 * np.pi)
    modes = np.arange(1, f_hz.size + 1)
    fig, ax = new_axes(xlabel="mode", ylabel="f [Hz]", title="Natural frequencies")
    ax.plot(modes, f_hz, "o", ms=5)
    return fig, ax
