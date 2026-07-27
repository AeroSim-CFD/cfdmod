"""Base (global) load reduction: overturning moments and their lever arms.

The failure these pin: a per-direction envelope in which the ``Mx`` panel is a
rescaled copy of the ``Fy`` panel. That happens when the base moment is formed
without a real lever arm (or with a constant one), and it is invisible by
inspection because a scaled copy still "looks like a moment". So the tests
below assert the three things a scaled copy cannot satisfy:

* the right-hand-rule signs (``Mx = -sum Fy z``, ``My = +sum Fx z``);
* that two load profiles with the *same* base force but different height
  distributions give *different* base moments;
* that a zero / missing lever ladder is an error, not a silently zeroed moment.
"""

from __future__ import annotations

import numpy as np
import pytest

from cfdmod.adapters.memory import MemoryFieldStore
from cfdmod.core import ElementMeta, PointsDataSource, TimeAxis, Topology
from cfdmod.core.container import Container
from cfdmod.dynamics import (
    BuildingCaseParameters,
    get_global_peaks_by_direction,
    global_load_history,
)

pytestmark = pytest.mark.unit

DT = 0.1


def _response(feq_x, feq_y, meq_z, z):
    fields = {
        "feq_x": np.asarray(feq_x, dtype=np.float64),
        "feq_y": np.asarray(feq_y, dtype=np.float64),
        "meq_z": np.asarray(meq_z, dtype=np.float64),
    }
    n_floors, n_t = fields["feq_x"].shape
    pts = np.zeros((n_floors, 3), dtype=np.float64)
    pts[:, 2] = np.asarray(z, dtype=np.float64)
    return PointsDataSource(
        time=TimeAxis(initial_time=0.0, timestep_size=DT, n_timesteps=n_t),
        topology=Topology.points(pts),
        elements=ElementMeta(position=pts),
        fields=MemoryFieldStore(fields),
    )


def _zeros(n_floors, n_t):
    return np.zeros((n_floors, n_t), dtype=np.float64)


# -- signs -----------------------------------------------------------------


def test_unit_force_at_one_height_gives_the_textbook_moment():
    """m = r x F with r = (0, 0, z), F = (Fx, Fy, 0) -> (-z*Fy, +z*Fx, 0)."""
    z = [4.0, 25.0]
    fx = np.array([[0.0], [1.0]])  # 1 N at z = 25
    resp = _response(fx, _zeros(2, 1), _zeros(2, 1), z)
    forces, moments = global_load_history(resp)
    assert forces["x"][0] == pytest.approx(1.0)
    assert moments["y"][0] == pytest.approx(25.0)  # My = +z * Fx
    assert moments["x"][0] == pytest.approx(0.0)

    fy = np.array([[0.0], [1.0]])
    resp = _response(_zeros(2, 1), fy, _zeros(2, 1), z)
    forces, moments = global_load_history(resp)
    assert forces["y"][0] == pytest.approx(1.0)
    assert moments["x"][0] == pytest.approx(-25.0)  # Mx = -z * Fy
    assert moments["y"][0] == pytest.approx(0.0)


def test_overturning_moment_opposes_the_transverse_force_sign():
    """A positive Fy must produce a negative Mx; the two panels are mirrored."""
    z = [10.0, 20.0, 30.0]
    fy = np.array([[1.0, -1.0], [2.0, -2.0], [3.0, -3.0]])
    resp = _response(_zeros(3, 2), fy, _zeros(3, 2), z)
    forces, moments = global_load_history(resp)
    assert np.all(np.sign(moments["x"]) == -np.sign(forces["y"]))


def test_torsion_needs_no_lever_arm():
    """meq_z is already a moment; it is summed straight through."""
    z = [10.0, 20.0]
    mz = np.array([[2.0, -2.0], [1.0, 5.0]])
    resp = _response(_zeros(2, 2), _zeros(2, 2), mz, z)
    _, moments = global_load_history(resp)
    np.testing.assert_allclose(moments["z"], [3.0, 3.0])


# -- the lever arm actually does work --------------------------------------


def test_same_base_force_different_height_distribution_gives_different_moment():
    """The guard against "Mx is a rescaled Fy".

    Both profiles have base ``Fy = 2``. One puts the load low, the other high;
    a moment computed with a real lever arm must separate them, a rescaled copy
    of the force cannot.
    """
    z = [10.0, 90.0]
    low = _response(_zeros(2, 1), np.array([[2.0], [0.0]]), _zeros(2, 1), z)
    high = _response(_zeros(2, 1), np.array([[0.0], [2.0]]), _zeros(2, 1), z)

    f_low, m_low = global_load_history(low)
    f_high, m_high = global_load_history(high)

    assert f_low["y"][0] == pytest.approx(f_high["y"][0])  # identical base force
    assert m_low["x"][0] == pytest.approx(-20.0)
    assert m_high["x"][0] == pytest.approx(-180.0)
    assert abs(m_high["x"][0]) > 5 * abs(m_low["x"][0])


def test_moment_is_not_proportional_to_force_across_directions():
    """Across a direction set, Mx / Fy must not be a single constant.

    If it is, the base moment carries no distribution information and the
    ``Mx`` panel of the directional envelope is a scaled copy of ``Fy``.
    """
    z = np.array([10.0, 50.0, 90.0])
    profiles = {
        0.0: np.array([[3.0], [1.0], [0.0]]),  # bottom-heavy
        90.0: np.array([[0.0], [1.0], [3.0]]),  # top-heavy
        180.0: np.array([[1.0], [2.0], [1.0]]),  # mid-heavy
    }
    container = Container(
        items={
            BuildingCaseParameters(direction=d, xi=0.0, recurrence_period=50.0): _response(
                _zeros(3, 1), fy, _zeros(3, 1), z
            )
            for d, fy in profiles.items()
        }
    )
    frames = get_global_peaks_by_direction(container, variable_type="static")
    fy_mean = frames["forces_static"]["mean_y"].to_numpy()
    mx_mean = frames["moments_static"]["mean_x"].to_numpy()

    ratios = mx_mean / fy_mean
    assert np.ptp(ratios) > 1e-6, "Mx / Fy is constant across directions -- no lever arm"


# -- lever arms must exist -------------------------------------------------


def test_all_zero_lever_arms_raise_instead_of_zeroing_the_moments():
    resp = _response(np.ones((2, 3)), np.ones((2, 3)), np.ones((2, 3)), [0.0, 0.0])
    with pytest.raises(ValueError, match="all zero"):
        global_load_history(resp)


def test_lever_heights_length_is_validated():
    resp = _response(np.ones((2, 3)), np.ones((2, 3)), np.ones((2, 3)), [1.0, 2.0])
    with pytest.raises(ValueError, match="must have 2 entries"):
        global_load_history(resp, lever_heights=np.array([1.0, 2.0, 3.0]))


def test_explicit_lever_heights_override_the_positions():
    resp = _response(np.array([[1.0]]), _zeros(1, 1), _zeros(1, 1), [1.0])
    _, moments = global_load_history(resp, lever_heights=np.array([7.0]))
    assert moments["y"][0] == pytest.approx(7.0)


# -- producer / consumer contract ------------------------------------------


@pytest.mark.parametrize(
    ("variable_type", "keys"),
    [
        ("static", ("forces_static", "moments_static")),
        ("hfpi", ("forces_static_eq", "moments_static_eq")),
    ],
)
def test_frames_match_what_the_plotter_consumes(variable_type, keys):
    """plot_global_stats_per_direction draws Fx, Fy, Mx, My, Mz off these frames."""
    plotting = pytest.importorskip("cfdmod.dynamics.plotting")
    matplotlib = pytest.importorskip("matplotlib")
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    z = [10.0, 20.0]
    container = Container(
        items={
            BuildingCaseParameters(direction=d, xi=0.0, recurrence_period=50.0): _response(
                np.array([[1.0, 2.0], [3.0, 4.0]]),
                np.array([[0.5, -0.5], [1.0, -1.0]]),
                np.array([[2.0, 1.0], [0.0, 3.0]]),
                z,
            )
            for d in (0.0, 90.0, 180.0)
        }
    )
    frames = get_global_peaks_by_direction(container, variable_type=variable_type)
    assert set(frames) == set(keys)
    for frame in frames.values():
        for stat in ("min", "max", "mean"):
            for axis in ("x", "y", "z"):
                assert f"{stat}_{axis}" in frame.columns

    stats = dict(frames)
    if variable_type == "hfpi":
        # the plotter reads the static frames for the axis setup regardless
        stats |= {
            "forces_static": frames["forces_static_eq"],
            "moments_static": frames["moments_static_eq"],
        }
    fig, _ = plotting.plot_global_stats_per_direction({0.0: stats}, variable_types=[variable_type])
    plt.close(fig)
