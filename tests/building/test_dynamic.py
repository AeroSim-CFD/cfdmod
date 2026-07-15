"""Tests for cfdmod.building.dynamic (floor loads + structure + response).

Covers the recipe wiring that turns per-floor Cf/Cm into a floor-load source,
synthesises a demo structural model, runs the modal response and reduces it to
the engineer-facing peak table. Inputs are small synthetic in-memory sources
(no fixtures / no mesh), so the module's own glue is exercised independently of
the lower-level dynamic recipe (pinned separately in
tests/core/recipes/test_building_dynamic.py).
"""

from __future__ import annotations

import numpy as np
import pytest

from cfdmod.adapters.memory import MemoryFieldStore
from cfdmod.core import ElementMeta, PointsDataSource, TimeAxis, Topology

building = pytest.importorskip("cfdmod.building")

pytestmark = pytest.mark.unit

N_FLOORS = 3
N_T = 120
DT = 0.05


def _case(**overrides):
    kw = dict(
        name="t",
        reference_height=30.0,
        characteristic_length=10.0,
        basic_wind_speed=40.0,
        simul_reference_velocity=40.0,
        nominal_area=100.0,
        nominal_volume=3000.0,
        floor_heights=[0.0, 10.0, 20.0, 30.0],  # -> 3 floors, mid-heights 5/15/25
    )
    kw.update(overrides)
    return building.BuildingCase(**kw)


def _coef_source(cf_x, cf_y, cm_z, *, n_floors=N_FLOORS, n_t=N_T):
    pts = np.zeros((n_floors, 3), dtype=np.float64)
    return PointsDataSource(
        time=TimeAxis(initial_time=0.0, timestep_size=DT, n_timesteps=n_t),
        topology=Topology.points(pts),
        elements=ElementMeta(position=pts),
        fields=MemoryFieldStore(
            {
                "cf_x": np.asarray(cf_x, dtype=np.float64),
                "cf_y": np.asarray(cf_y, dtype=np.float64),
                "cm_z": np.asarray(cm_z, dtype=np.float64),
            }
        ),
    )


def _uniform_coefs(vx, vy, vz, *, n_floors=N_FLOORS, n_t=N_T):
    ones = np.ones((n_floors, n_t), dtype=np.float64)
    return _coef_source(vx * ones, vy * ones, vz * ones, n_floors=n_floors, n_t=n_t)


# -- floor_load_source -----------------------------------------------------


def test_floor_load_source_fields_and_shape():
    case = _case()
    src = _uniform_coefs(1.0, 2.0, 3.0)
    load = building.floor_load_source(src, src, case)
    assert load.kind == "points"
    assert load.n_elements == N_FLOORS
    for f in ("cf_x", "cf_y", "cm_z"):
        arr = np.asarray(load.fields.read(f))
        assert arr.shape == (N_FLOORS, N_T)
        assert np.isfinite(arr).all()


def test_floor_load_source_dimensionalizes_with_q_area_volume():
    """F = cf * q * A, M = cm_z * q * V with q = 0.5 rho U_H^2."""
    case = _case()
    q = case.dynamic_pressure  # 0.5 * 1.225 * 40^2 = 980
    src = _uniform_coefs(1.0, 2.0, 3.0)
    load = building.floor_load_source(src, src, case, dimensionalize=True)
    np.testing.assert_allclose(load.fields.read("cf_x"), 1.0 * q * case.nominal_area)
    np.testing.assert_allclose(load.fields.read("cf_y"), 2.0 * q * case.nominal_area)
    np.testing.assert_allclose(load.fields.read("cm_z"), 3.0 * q * case.nominal_volume)


def test_floor_load_source_raw_coefficients_when_not_dimensionalized():
    case = _case()
    src = _uniform_coefs(1.0, 2.0, 3.0)
    load = building.floor_load_source(src, src, case, dimensionalize=False)
    np.testing.assert_allclose(load.fields.read("cf_x"), 1.0)
    np.testing.assert_allclose(load.fields.read("cf_y"), 2.0)
    np.testing.assert_allclose(load.fields.read("cm_z"), 3.0)


def test_floor_load_source_rejects_mismatched_floor_counts():
    case = _case()
    cf = _uniform_coefs(1.0, 1.0, 1.0, n_floors=N_FLOORS)
    cm = _uniform_coefs(1.0, 1.0, 1.0, n_floors=N_FLOORS + 1)
    with pytest.raises(ValueError, match="floor counts disagree"):
        building.floor_load_source(cf, cm, case)


# -- _floor_mid_heights ----------------------------------------------------


def test_floor_mid_heights_from_edges():
    from cfdmod.building import dynamic

    case = _case()  # edges 0/10/20/30
    mids = dynamic._floor_mid_heights(case, N_FLOORS)
    np.testing.assert_allclose(mids, [5.0, 15.0, 25.0])


def test_floor_mid_heights_falls_back_to_integer_ladder_on_count_mismatch():
    from cfdmod.building import dynamic

    case = _case()  # 4 edges -> supports 3 floors only
    mids = dynamic._floor_mid_heights(case, 5)  # pressure dropped/added floors
    np.testing.assert_allclose(mids, [0.0, 1.0, 2.0, 3.0, 4.0])


# -- example_building_structure --------------------------------------------


def test_example_structure_shapes_and_modes():
    case = _case()
    st = building.example_building_structure(case, N_FLOORS)
    assert st.n_floors == N_FLOORS
    assert st.n_modes == 3
    assert np.asarray(st.mode_shapes).shape == (N_FLOORS, 3, 3)
    assert np.isfinite(np.asarray(st.mode_shapes)).all()


@pytest.mark.parametrize("requested,expected", [(1, 1), (2, 2), (3, 3), (5, 3), (0, 1)])
def test_example_structure_clamps_n_modes(requested, expected):
    case = _case()
    st = building.example_building_structure(case, N_FLOORS, n_modes=requested)
    assert st.n_modes == expected


def test_example_structure_default_frequency_is_ellis():
    """f1 defaults to the Ellis 46/H fundamental (stored as angular wp)."""
    case = _case(reference_height=30.0)
    st = building.example_building_structure(case, N_FLOORS)
    wp = np.asarray(st.natural_frequencies, dtype=np.float64)
    f_hz = wp / (2.0 * np.pi)
    assert f_hz[0] == pytest.approx(46.0 / 30.0, rel=1e-6)
    # higher modes at 1.1x / 1.25x the fundamental
    assert f_hz[1] == pytest.approx(46.0 / 30.0 * 1.1, rel=1e-6)
    assert f_hz[2] == pytest.approx(46.0 / 30.0 * 1.25, rel=1e-6)


def test_example_structure_explicit_floor_mass_overrides_scaling():
    case = _case()
    st = building.example_building_structure(case, N_FLOORS, floor_mass=12345.0)
    np.testing.assert_allclose(st.floors_mass, 12345.0)


# -- solve + accelerations + peak table (end to end) -----------------------


@pytest.fixture(scope="module")
def response_bundle():
    """A finite dynamic response + accelerations on a smooth synthetic load."""
    case = _case()
    t = np.arange(N_T) * DT

    def series(scale, phase):
        base = np.sin(2 * np.pi * 0.3 * t + phase) + 0.4 * np.sin(2 * np.pi * 0.7 * t)
        return np.vstack([scale * (1 + 0.1 * f) * base for f in range(N_FLOORS)])

    src = _coef_source(series(1.0, 0.1), series(0.7, 0.5), series(0.2, 1.0))
    load = building.floor_load_source(src, src, case)
    structure = building.example_building_structure(case, load.n_elements)
    response = building.solve_building_response(load, structure, damping_ratio=0.02)
    acc = building.floor_accelerations(response, structure, point=(1.0, 0.0))
    return case, structure, response, acc


def test_solve_building_response_finite_expected_shape(response_bundle):
    _, _, response, _ = response_bundle
    for name in ("disp_x", "disp_y", "rot_z", "feq_x", "feq_y", "meq_z"):
        arr = np.asarray(response.fields.read(name))
        assert arr.shape == (N_FLOORS, N_T)
        assert np.isfinite(arr).all()


def test_floor_accelerations_finite(response_bundle):
    _, _, _, acc = response_bundle
    for name in ("acc_x", "acc_y", "acc_mag"):
        arr = np.asarray(acc.fields.read(name))
        assert arr.shape == (N_FLOORS, N_T)
        assert np.isfinite(arr).all()
    # magnitude is the non-negative resultant of the two components
    assert np.all(np.asarray(acc.fields.read("acc_mag")) >= 0.0)


def test_peak_response_table_one_row_per_floor(response_bundle):
    case, _, response, acc = response_bundle
    table = building.peak_response_table(response, acc, case)
    assert len(table) == N_FLOORS
    expected_cols = {
        "floor",
        "z_mid",
        "disp_x_peak",
        "disp_y_peak",
        "rot_z_peak",
        "feq_x_peak",
        "feq_y_peak",
        "meq_z_peak",
        "acc_mag_peak",
    }
    assert expected_cols.issubset(table.columns)
    np.testing.assert_allclose(table["z_mid"], [5.0, 15.0, 25.0])
    # peaks are max-abs over time, hence non-negative and finite
    peak_cols = [c for c in table.columns if c.endswith("_peak")]
    assert (table[peak_cols].to_numpy() >= 0.0).all()
    assert np.isfinite(table[peak_cols].to_numpy()).all()
