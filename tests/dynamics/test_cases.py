"""Directional-case orchestration on Container + multiplier parity."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from cfdmod.adapters.memory import MemoryFieldStore
from cfdmod.building.peaks import peak_value
from cfdmod.core import ElementMeta, PointsDataSource, TimeAxis, Topology
from cfdmod.core.container import Container
from cfdmod.core.recipes import build_building_dynamic_response
from cfdmod.dynamics import (
    BuildingCaseParameters,
    build_cases,
    filter_by_kd,
    filter_by_recurrence_period,
    filter_by_xi,
    get_global_peaks_by_direction,
    get_max_acceleration,
    get_max_acceleration_by_recurrence_period,
    get_stats_forces_effective,
    join_by_direction,
    solve_building_cases,
)
from cfdmod.dynamics.structural import BuildingStructuralData, mass_normalize_mode_shapes

N_FLOORS = 3
N_MODES = 2
N_T = 200
DT = 0.05


def test_build_cases_cartesian_product():
    cases = build_cases(
        directions=[0.0, 90.0],
        xis=[0.01, 0.02],
        recurrence_periods=[50.0],
    )
    assert len(cases) == 4
    assert all(isinstance(c, BuildingCaseParameters) for c in cases)


def test_container_join_and_filter_by_direction_and_xi():
    cases = build_cases(directions=[0.0, 90.0, 180.0], xis=[0.01, 0.02], recurrence_periods=[50.0])

    # Trivial solve_fn returns an empty-ish points source tagged by nothing.
    def solve_fn(_c):
        pts = np.zeros((1, 3))
        return PointsDataSource(
            time=TimeAxis(initial_time=0.0, timestep_size=1.0, n_timesteps=1),
            topology=Topology.points(pts),
            elements=ElementMeta(position=pts),
            fields=MemoryFieldStore({"u": np.zeros((1, 1))}),
        )

    container = solve_building_cases(cases, solve_fn)
    assert len(container) == 6

    by_direction = container.join_by(lambda c: c.direction)
    assert set(by_direction.keys()) == {0.0, 90.0, 180.0}
    assert all(len(sub) == 2 for sub in by_direction.values())

    only_light_damping = container.filter_by(lambda c: c.xi == 0.01)
    assert len(only_light_damping) == 3


def _raw_inputs():
    t = np.arange(N_T) * DT

    def series(scale, phase):
        return scale * (np.sin(2 * np.pi * 0.3 * t + phase) + 0.3 * np.sin(2 * np.pi * 0.9 * t))

    cf_x = np.vstack([series(1.0 + 0.1 * f, 0.2 * f) for f in range(N_FLOORS)])
    cf_y = np.vstack([series(0.6, 0.4 * f) for f in range(N_FLOORS)])
    cm_z = np.vstack([series(0.2, 0.7 * f) for f in range(N_FLOORS)])

    heights = np.arange(1, N_FLOORS + 1) * 3.0
    df_floors = pd.DataFrame(
        {
            "Z": heights,
            "M": np.full(N_FLOORS, 110.0),
            "I": np.full(N_FLOORS, 950.0),
            "XR": np.full(N_FLOORS, 0.5),
            "YR": np.full(N_FLOORS, 0.2),
        }
    )
    df_floors["R"] = (df_floors["I"] / df_floors["M"]) ** 0.5

    df_modes = pd.DataFrame({"mode": [1, 2], "period": [1.0, 0.4]})
    df_modes["frequency"] = 1 / df_modes["period"]
    df_modes["wp"] = 2 * np.pi * df_modes["frequency"]

    shapes = [
        pd.DataFrame(
            {
                "DX": np.linspace(0.1, 1.0, N_FLOORS) * (1 + 0.2 * m),
                "DY": np.linspace(0.05, 0.5, N_FLOORS),
                "RZ": np.linspace(0.01, 0.03, N_FLOORS),
            }
        )
        for m in range(N_MODES)
    ]
    return cf_x, cf_y, cm_z, df_floors, df_modes, shapes


def _floor_source(cf_x, cf_y, cm_z):
    pts = np.zeros((N_FLOORS, 3))
    return PointsDataSource(
        time=TimeAxis(initial_time=0.0, timestep_size=DT, n_timesteps=N_T),
        topology=Topology.points(pts),
        elements=ElementMeta(position=pts),
        fields=MemoryFieldStore({"cf_x": cf_x, "cf_y": cf_y, "cm_z": cm_z}),
    )


def test_multiplier_parity_with_legacy_case_build():
    """with_multipliers(mm, fm) reproduces the frozen legacy case-knob outputs."""
    from tests.dynamics._goldens import golden

    cf_x, cf_y, cm_z, df_floors, df_modes, shapes = _raw_inputs()
    mm, fm = 1.4, 1.2
    xi = 0.015

    # base structural data (mm=1, fm=1), then with_multipliers
    phi_raw = np.stack([np.column_stack([s["DX"], s["DY"], s["RZ"]]) for s in shapes], axis=1)
    base_mass = df_floors["M"].to_numpy()
    base_radius = df_floors["R"].to_numpy()
    phi_norm = mass_normalize_mode_shapes(phi_raw, base_mass, base_radius)
    base = BuildingStructuralData(
        mode_shapes=phi_norm,
        natural_frequencies=df_modes["wp"].to_numpy(),
        floor_points=np.column_stack([np.zeros(N_FLOORS), np.zeros(N_FLOORS), df_floors["Z"]]),
        cm_positions=df_floors[["XR", "YR"]].to_numpy(),
        floors_mass=base_mass,
        floors_radius=base_radius,
    )
    case = base.with_multipliers(mass_multiplier=mm, frequency_multiplier=fm)
    out = build_building_dynamic_response(_floor_source(cf_x, cf_y, cm_z), case.to_config(xi))

    np.testing.assert_allclose(
        out.fields.read("disp_x"), golden("mul_disp_x"), rtol=1e-6, atol=1e-9
    )
    np.testing.assert_allclose(out.fields.read("feq_x"), golden("mul_feq_x"), rtol=1e-6, atol=1e-9)
    np.testing.assert_allclose(out.fields.read("meq_z"), golden("mul_meq_z"), rtol=1e-6, atol=1e-9)


# --- Multi-direction result queries (#177) ----------------------------------


def _response(fields: dict[str, np.ndarray]) -> PointsDataSource:
    """Build a floor-response PointsDataSource carrying the given fields."""
    n_t = next(iter(fields.values())).shape[1]
    n_floors = next(iter(fields.values())).shape[0]
    pts = np.zeros((n_floors, 3))
    return PointsDataSource(
        time=TimeAxis(initial_time=0.0, timestep_size=DT, n_timesteps=n_t),
        topology=Topology.points(pts),
        elements=ElementMeta(position=pts),
        fields=MemoryFieldStore({k: np.asarray(v, dtype=float) for k, v in fields.items()}),
    )


def test_filter_wrappers_match_container_primitives():
    cases = build_cases(
        directions=[0.0, 90.0, 180.0],
        xis=[0.01, 0.02],
        recurrence_periods=[10.0, 50.0],
        use_kd=[False, True],
    )

    def solve_fn(case):
        return _response({"acc_mag": np.full((N_FLOORS, N_T), case.recurrence_period)})

    container = solve_building_cases(cases, solve_fn)
    assert len(container) == 3 * 2 * 2 * 2

    # each wrapper is exactly the underlying Container.filter_by on that field
    assert set(filter_by_xi(container, 0.01).keys()) == set(
        container.filter_by(lambda k: k.xi == 0.01).keys()
    )
    assert len(filter_by_xi(container, 0.01)) == 12
    assert len(filter_by_kd(container, True)) == 12
    assert len(filter_by_recurrence_period(container, 10.0)) == 12

    by_direction = join_by_direction(container)
    assert set(by_direction.keys()) == {0.0, 90.0, 180.0}
    assert all(len(sub) == 8 for sub in by_direction.values())

    # absent value yields an empty sub-container (not a KeyError)
    assert len(filter_by_xi(container, 0.99)) == 0


def test_get_max_acceleration_and_by_recurrence_period():
    cases = build_cases(
        directions=[0.0, 90.0],
        xis=[0.01],
        recurrence_periods=[10.0, 50.0],
    )

    def solve_fn(case):
        # top-floor acceleration magnitude is a constant equal to the rec. period
        return _response({"acc_mag": np.full((N_FLOORS, N_T), case.recurrence_period)})

    container = solve_building_cases(cases, solve_fn)
    assert get_max_acceleration(container, method="max") == pytest.approx(50.0)
    assert get_max_acceleration_by_recurrence_period(container, method="max") == {
        10.0: pytest.approx(10.0),
        50.0: pytest.approx(50.0),
    }


def test_get_max_acceleration_wires_peak_methods():
    t = np.arange(N_T) * DT
    series = 2.0 * np.sin(2 * np.pi * 0.3 * t) + 0.5 * np.sin(2 * np.pi * 0.9 * t)
    acc_mag = np.zeros((2, N_T))
    acc_mag[-1] = series  # top floor carries the known series

    container = solve_building_cases(
        build_cases(directions=[0.0], xis=[0.01], recurrence_periods=[10.0]),
        lambda _c: _response({"acc_mag": acc_mag}),
    )

    for method in ("max", "gumbel"):
        assert get_max_acceleration(container, method=method) == pytest.approx(
            peak_value(series, method, absolute=True)
        )
    assert get_max_acceleration(container, method="peak-factor", f0=0.3) == pytest.approx(
        peak_value(series, "peak-factor", absolute=True, f0=0.3)
    )


def test_get_stats_forces_effective_signed_extrema():
    feq_x = np.array([[1.0, -3.0], [2.0, -2.0]])  # (n_floors=2, n_t=2)
    feq_y = np.array([[4.0, 0.0], [-1.0, 5.0]])
    meq_z = np.array([[10.0, 20.0], [-5.0, -15.0]])
    resp = _response({"feq_x": feq_x, "feq_y": feq_y, "meq_z": meq_z})

    s_max = get_stats_forces_effective(resp, "max")
    np.testing.assert_allclose(s_max["x"], [1.0, 2.0])
    np.testing.assert_allclose(s_max["y"], [4.0, 5.0])
    np.testing.assert_allclose(s_max["z"], [20.0, -5.0])

    s_min = get_stats_forces_effective(resp, "min")
    np.testing.assert_allclose(s_min["x"], [-3.0, -2.0])
    np.testing.assert_allclose(s_min["z"], [10.0, -15.0])

    s_mean = get_stats_forces_effective(resp, "mean")
    np.testing.assert_allclose(s_mean["x"], [-1.0, 0.0])

    with pytest.raises(ValueError):
        get_stats_forces_effective(resp, "median")  # type: ignore[arg-type]


def test_get_global_peaks_by_direction():
    feq_x = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])  # sum over floors -> [5, 7, 9]
    feq_y = np.array([[0.0, 0.0, 0.0], [1.0, 1.0, 1.0]])  # -> [1, 1, 1]
    meq_z = np.array([[2.0, -2.0, 0.0], [0.0, 0.0, 0.0]])  # -> [2, -2, 0]

    def solve_fn(case):
        scale = 1.0 if case.direction == 0.0 else 10.0
        return _response({"feq_x": scale * feq_x, "feq_y": scale * feq_y, "meq_z": scale * meq_z})

    container = solve_building_cases(
        build_cases(directions=[90.0, 0.0], xis=[0.01], recurrence_periods=[10.0]),
        solve_fn,
    )
    frames = get_global_peaks_by_direction(container)
    forces = frames["forces_static_eq"]
    moments = frames["moments_static_eq"]

    assert list(forces["direction"]) == [0.0, 90.0]  # sorted by direction
    assert set(forces.columns) == {
        "direction",
        "min_x",
        "max_x",
        "mean_x",
        "min_y",
        "max_y",
        "mean_y",
    }
    assert set(moments.columns) == {"direction", "min_z", "max_z", "mean_z"}

    row0 = forces.iloc[0]
    assert (row0["min_x"], row0["max_x"], row0["mean_x"]) == pytest.approx((5.0, 9.0, 7.0))
    assert moments.iloc[0]["min_z"] == pytest.approx(-2.0)
    # direction 90 is the x10 case
    assert forces.iloc[1]["max_x"] == pytest.approx(90.0)

    # two cases at the same direction is ambiguous
    two_same_dir = Container(
        items={
            BuildingCaseParameters(direction=0.0, xi=0.01, recurrence_period=10.0): _response(
                {"feq_x": feq_x, "feq_y": feq_y, "meq_z": meq_z}
            ),
            BuildingCaseParameters(direction=0.0, xi=0.02, recurrence_period=10.0): _response(
                {"feq_x": feq_x, "feq_y": feq_y, "meq_z": meq_z}
            ),
        }
    )
    with pytest.raises(ValueError):
        get_global_peaks_by_direction(two_same_dir)
