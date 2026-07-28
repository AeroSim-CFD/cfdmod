"""The dynamic stage must complain when its load time axis cannot carry the modes.

A load record left on the solver's normalised time, or de-normalised with the
wrong characteristic length, still has the right shape, the right units and a
plausible range -- nothing downstream can tell. These two cheap guards catch the
gross cases at the only place that knows both the time axis and the natural
frequencies.
"""

from __future__ import annotations

import warnings

import numpy as np
import pytest

from cfdmod.adapters.memory import MemoryFieldStore
from cfdmod.building import solve_building_response
from cfdmod.core import ElementMeta, PointsDataSource, TimeAxis, Topology
from cfdmod.core.recipes import check_modal_sampling
from cfdmod.dynamics import BuildingStructuralData, mass_normalize_mode_shapes

N_FLOORS = 6


def _axis(dt: float, n_t: int) -> TimeAxis:
    return TimeAxis(initial_time=0.0, timestep_size=dt, n_timesteps=n_t)


def _load(dt: float, n_t: int) -> PointsDataSource:
    rng = np.random.default_rng(0)
    pts = np.column_stack(
        [np.zeros(N_FLOORS), np.zeros(N_FLOORS), np.linspace(5.0, 60.0, N_FLOORS)]
    )
    fields = {k: rng.standard_normal((N_FLOORS, n_t)) * 1e4 for k in ("cf_x", "cf_y", "cm_z")}
    return PointsDataSource(
        time=_axis(dt, n_t),
        topology=Topology.points(pts),
        elements=ElementMeta(position=pts),
        fields=MemoryFieldStore(fields),
    )


def _structure() -> BuildingStructuralData:
    """Two sway modes on a plain 6-floor cantilever, f = 0.25 / 0.9 Hz."""
    z = np.linspace(5.0, 60.0, N_FLOORS)
    z_norm = z / z.max()
    phi = np.stack(
        [
            np.column_stack([z_norm, np.zeros(N_FLOORS), np.zeros(N_FLOORS)]),
            np.column_stack([np.zeros(N_FLOORS), z_norm, np.zeros(N_FLOORS)]),
        ],
        axis=1,
    )
    mass = np.full(N_FLOORS, 1.0e6)
    radius = np.full(N_FLOORS, 10.0)
    return BuildingStructuralData(
        mode_shapes=mass_normalize_mode_shapes(phi, mass, radius),
        natural_frequencies=2 * np.pi * np.array([0.25, 0.9]),
        floor_points=np.column_stack([np.zeros(N_FLOORS), np.zeros(N_FLOORS), z]),
        cm_positions=np.zeros((N_FLOORS, 2)),
        floors_mass=mass,
        floors_radius=radius,
    )


@pytest.mark.unit
def test_well_sampled_axis_is_silent():
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        check_modal_sampling(_axis(0.05, 20000), [2 * np.pi * 0.2, 2 * np.pi * 1.0])


@pytest.mark.unit
def test_warns_when_the_highest_mode_is_undersampled():
    with pytest.warns(UserWarning, match="points per cycle"):
        check_modal_sampling(_axis(0.5, 20000), [2 * np.pi * 0.2, 2 * np.pi * 1.5])


@pytest.mark.unit
def test_warns_when_the_record_is_too_short():
    with pytest.warns(UserWarning, match="cycles of the fundamental"):
        check_modal_sampling(_axis(0.01, 500), [2 * np.pi * 0.2])


@pytest.mark.unit
def test_solve_building_response_warns_on_a_normalised_time_axis():
    """dt still in convective units (~0.02) is ~50x too small -> short record."""
    with pytest.warns(UserWarning):
        solve_building_response(_load(0.02, 400), _structure(), damping_ratio=0.02)


@pytest.mark.unit
def test_sampling_check_can_be_disabled():
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        solve_building_response(
            _load(0.02, 400), _structure(), damping_ratio=0.02, check_sampling=False
        )
