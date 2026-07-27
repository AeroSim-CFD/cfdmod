"""Tests for cfdmod.building.static (static-equivalent floor loads).

These pin the two properties a structural deliverable depends on and that a
hand-rolled notebook cell got wrong on a real consulting case:

1. The loads are referenced at an **explicitly supplied** design speed, not at
   the case's simulation inlet speed. ``BuildingCase.dynamic_pressure`` is built
   from ``simul_reference_velocity`` (what Cp was normalised by); using it for
   the deliverable scales every load by ``(U_simul / U_design)^2``, which on a
   real tower ranges from ~0.8 to ~1.5 and *varies per wind direction*, so it
   distorts the directional envelope as well as the magnitude.
2. The floor points carry their lever-arm heights, so the base overturning
   moments downstream (:func:`cfdmod.dynamics.global_load_history`) have an arm
   to work with instead of a zero ladder.
"""

from __future__ import annotations

import numpy as np
import pytest

from cfdmod.adapters.memory import MemoryFieldStore
from cfdmod.core import ElementMeta, PointsDataSource, TimeAxis, Topology

building = pytest.importorskip("cfdmod.building")

pytestmark = pytest.mark.unit

N_FLOORS = 3
N_T = 20
DT = 0.05
EDGES = [0.0, 10.0, 20.0, 30.0]  # 3 floors; tops 10/20/30, mids 5/15/25


def _case(**overrides):
    kw = dict(
        name="t",
        reference_height=30.0,
        characteristic_length=10.0,
        basic_wind_speed=40.0,
        simul_reference_velocity=40.0,
        nominal_area=100.0,
        nominal_volume=3000.0,
        floor_heights=list(EDGES),
        fluid_density=1.225,
    )
    kw.update(overrides)
    return building.BuildingCase(**kw)


def _coefs(vx, vy, vz, *, n_floors=N_FLOORS, n_t=N_T):
    ones = np.ones((n_floors, n_t), dtype=np.float64)
    pts = np.zeros((n_floors, 3), dtype=np.float64)
    return PointsDataSource(
        time=TimeAxis(initial_time=0.0, timestep_size=DT, n_timesteps=n_t),
        topology=Topology.points(pts),
        elements=ElementMeta(position=pts),
        fields=MemoryFieldStore({"cf_x": vx * ones, "cf_y": vy * ones, "cm_z": vz * ones}),
    )


# -- floor_lever_heights ---------------------------------------------------


def test_floor_lever_heights_conventions():
    assert building.floor_lever_heights(EDGES, 3, lever="top").tolist() == [10.0, 20.0, 30.0]
    assert building.floor_lever_heights(EDGES, 3, lever="bottom").tolist() == [0.0, 10.0, 20.0]
    assert building.floor_lever_heights(EDGES, 3, lever="mid").tolist() == [5.0, 15.0, 25.0]


def test_floor_lever_heights_rejects_mismatched_edges():
    """A silent index-ladder fallback would put lever arms in the wrong units."""
    with pytest.raises(ValueError, match="ascending z-edges"):
        building.floor_lever_heights(EDGES, 5)
    with pytest.raises(ValueError, match="unknown lever"):
        building.floor_lever_heights(EDGES, 3, lever="middle")  # type: ignore[arg-type]


# -- static_floor_loads ----------------------------------------------------


def test_static_floor_loads_uses_the_supplied_reference_velocity():
    """F = cf * q(U_load) * A -- q comes from the argument, never from the case."""
    case = _case(simul_reference_velocity=40.0)
    u_load = 30.0
    q = 0.5 * case.fluid_density * u_load**2
    loads = building.static_floor_loads(
        _coefs(1.0, 2.0, 3.0), _coefs(1.0, 2.0, 3.0), case, reference_velocity=u_load
    )
    np.testing.assert_allclose(loads.fields.read("feq_x"), 1.0 * q * case.nominal_area)
    np.testing.assert_allclose(loads.fields.read("feq_y"), 2.0 * q * case.nominal_area)
    np.testing.assert_allclose(loads.fields.read("meq_z"), 3.0 * q * case.nominal_volume)


def test_static_floor_loads_does_not_silently_use_the_simulation_speed():
    """The regression: loads at the design speed must differ from case.dynamic_pressure."""
    case = _case(simul_reference_velocity=29.699)  # what Cp was normalised by
    design_u_h = 36.528  # 50-year design U_H for this direction
    src = _coefs(1.0, 1.0, 1.0)

    at_design = np.asarray(
        building.static_floor_loads(src, src, case, reference_velocity=design_u_h).fields.read(
            "feq_x"
        )
    )
    at_simul = np.asarray(
        building.static_floor_loads(
            src, src, case, reference_velocity=case.simul_reference_velocity
        ).fields.read("feq_x")
    )

    expected_ratio = (design_u_h / case.simul_reference_velocity) ** 2
    np.testing.assert_allclose(at_design / at_simul, expected_ratio)
    assert expected_ratio == pytest.approx(1.513, rel=1e-3)


@pytest.mark.parametrize("bad", [0.0, -1.0, float("nan")])
def test_static_floor_loads_rejects_non_positive_reference_velocity(bad):
    case = _case()
    src = _coefs(1.0, 1.0, 1.0)
    with pytest.raises(ValueError, match="positive speed"):
        building.static_floor_loads(src, src, case, reference_velocity=bad)


def test_static_floor_loads_scales_quadratically_with_speed():
    case = _case()
    src = _coefs(1.0, 1.0, 1.0)
    a = np.asarray(
        building.static_floor_loads(src, src, case, reference_velocity=20.0).fields.read("feq_x")
    )
    b = np.asarray(
        building.static_floor_loads(src, src, case, reference_velocity=40.0).fields.read("feq_x")
    )
    np.testing.assert_allclose(b / a, 4.0)


def test_static_floor_loads_positions_carry_the_lever_arms():
    """Base overturning moments read elements.position[:, 2] -- it must not be zeros."""
    case = _case()
    src = _coefs(1.0, 1.0, 1.0)
    loads = building.static_floor_loads(src, src, case, reference_velocity=30.0)
    z = np.asarray(loads.elements.position)[:, 2]
    np.testing.assert_allclose(z, [10.0, 20.0, 30.0])
    assert np.any(z)

    mid = building.static_floor_loads(src, src, case, reference_velocity=30.0, lever="mid")
    np.testing.assert_allclose(np.asarray(mid.elements.position)[:, 2], [5.0, 15.0, 25.0])


def test_static_floor_loads_rejects_unalignable_floor_count():
    """Fewer rows than bands is only OK when the row -> band mapping is recoverable."""
    case = _case(floor_heights=[0.0, 10.0, 20.0, 30.0, 40.0])  # 4 bands vs 3 coefficient rows
    src = _coefs(1.0, 1.0, 1.0)  # plain points source: no floor grouping to read
    with pytest.raises(ValueError, match="could not be recovered"):
        building.static_floor_loads(src, src, case, reference_velocity=30.0)


def test_scatter_to_floor_bands_places_rows_by_index():
    """Empty *interior* bands must not shift the profile -- rows go where the index says."""
    rows = np.array([[1.0, 2.0], [3.0, 4.0]])
    full = building.scatter_to_floor_bands(rows, np.array([0, 3]), 5)
    assert full.shape == (5, 2)
    np.testing.assert_allclose(full[0], [1.0, 2.0])
    np.testing.assert_allclose(full[3], [3.0, 4.0])
    np.testing.assert_allclose(full[[1, 2, 4]], 0.0)

    with pytest.raises(ValueError, match="band indices"):
        building.scatter_to_floor_bands(rows, np.array([0, 9]), 5)
    with pytest.raises(ValueError, match="band indices for"):
        building.scatter_to_floor_bands(rows, np.array([0, 1, 2]), 5)


def test_static_floor_loads_rejects_disagreeing_coefficient_shapes():
    case = _case()
    cf = _coefs(1.0, 1.0, 1.0)
    cm = _coefs(1.0, 1.0, 1.0, n_floors=2)
    with pytest.raises(ValueError, match="floor counts disagree"):
        building.static_floor_loads(cf, cm, case, reference_velocity=30.0)


# -- floor_load_source (dynamic sibling) -----------------------------------


def test_floor_load_source_accepts_an_explicit_reference_velocity():
    case = _case(simul_reference_velocity=40.0)
    src = _coefs(1.0, 1.0, 1.0)
    default = np.asarray(building.floor_load_source(src, src, case).fields.read("cf_x"))
    explicit = np.asarray(
        building.floor_load_source(src, src, case, reference_velocity=20.0).fields.read("cf_x")
    )
    np.testing.assert_allclose(default / explicit, (40.0 / 20.0) ** 2)
