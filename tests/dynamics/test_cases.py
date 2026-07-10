"""Directional-case orchestration on Container + multiplier parity."""

from __future__ import annotations

import numpy as np
import pandas as pd

from cfdmod.adapters.memory import MemoryFieldStore
from cfdmod.core import ElementMeta, PointsDataSource, TimeAxis, Topology
from cfdmod.core.recipes import build_building_dynamic_response
from cfdmod.dynamics import (
    BuildingCaseParameters,
    build_cases,
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
