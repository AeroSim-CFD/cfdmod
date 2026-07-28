"""Point-acceleration (comfort) recipe: parity + peak reduction.

The acceleration *physics* (rotational lever arm + second time-derivative)
is pinned against the legacy ``HFPIResults.get_point_acceleration``. The peak
reduction reuses the shared ``extreme_value`` op; the Gumbel estimator there
deliberately differs from the legacy one (a documented bias fix), so only the
acceleration series is parity-tested, not the Gumbel quantile.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from cfdmod.adapters.memory import MemoryFieldStore
from cfdmod.core import ElementMeta, PointsDataSource, TimeAxis, Topology
from cfdmod.core.ops.data_source_create.extreme_value import (
    ExtremeValueParams,
    extreme_value,
)
from cfdmod.core.recipes import (
    BuildingDynamicConfig,
    ComfortConfig,
    build_building_dynamic_response,
    build_point_accelerations,
)

N_FLOORS = 3
N_MODES = 2
N_T = 240
DT = 0.05
POINT = (2.0, 1.0)


def _synthetic():
    t = np.arange(N_T) * DT

    def series(scale, phase):
        return scale * (np.sin(2 * np.pi * 0.3 * t + phase) + 0.4 * np.sin(2 * np.pi * 0.7 * t))

    cf_x = np.vstack([series(1.0 + 0.2 * f, 0.1 * f) for f in range(N_FLOORS)])
    cf_y = np.vstack([series(0.7, 0.5 + 0.1 * f) for f in range(N_FLOORS)])
    cm_z = np.vstack([series(0.2, 1.0 + 0.2 * f) for f in range(N_FLOORS)])

    df_floors = pd.DataFrame(
        {
            "Z": np.arange(1, N_FLOORS + 1) * 3.0,
            "M": np.full(N_FLOORS, 120.0),
            "I": np.full(N_FLOORS, 900.0),
            "XR": np.full(N_FLOORS, 0.5),
            "YR": np.full(N_FLOORS, 0.2),
        }
    )
    df_floors["R"] = (df_floors["I"] / df_floors["M"]) ** 0.5
    df_modes = pd.DataFrame({"mode": [1, 2], "period": [1.0, 0.5]})
    df_modes["frequency"] = 1 / df_modes["period"]
    df_modes["wp"] = 2 * np.pi * df_modes["frequency"]
    shapes = [
        pd.DataFrame(
            {
                "DX": np.linspace(0.1, 1.0, N_FLOORS) * (1 + 0.3 * m),
                "DY": np.linspace(0.05, 0.6, N_FLOORS),
                "RZ": np.linspace(0.01, 0.05, N_FLOORS),
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


def _nrms(actual, expected) -> float:
    """Normalized RMS difference, robust to the signals' zero crossings."""
    a = np.asarray(actual, dtype=np.float64)
    e = np.asarray(expected, dtype=np.float64)
    return float(np.sqrt(np.mean((a - e) ** 2)) / np.sqrt(np.mean(e**2)))


def test_acceleration_parity_with_legacy_point_acceleration():
    """Characterization against the legacy point-acceleration path.

    The goldens come from the legacy adaptive-RK45 solve at SciPy's default
    tolerances, whose own integration error on a lightly-damped modal response
    is ~1%. The recipe now solves the same ODE in closed form, so parity is
    asserted as a normalized RMS difference at a tolerance that admits that
    legacy error -- the analytical accuracy of the solver itself is pinned in
    tests/core/recipes/test_sdof_exact_solver.py.
    """
    from cfdmod.dynamics.structural import mass_normalize_mode_shapes
    from tests.dynamics._goldens import golden

    cf_x, cf_y, cm_z, df_floors, df_modes, shapes = _synthetic()

    phi_raw = np.stack([np.column_stack([s["DX"], s["DY"], s["RZ"]]) for s in shapes], axis=1)
    phi = mass_normalize_mode_shapes(phi_raw, df_floors["M"].to_numpy(), df_floors["R"].to_numpy())
    cfg = BuildingDynamicConfig(
        mode_shapes=phi,
        floor_points=np.column_stack([np.zeros(N_FLOORS), np.zeros(N_FLOORS), df_floors["Z"]]),
        cm_positions=df_floors[["XR", "YR"]].to_numpy(),
        floors_mass=df_floors["M"].to_numpy(),
        floors_radius=df_floors["R"].to_numpy(),
        natural_frequencies=df_modes["wp"].to_numpy(),
        damping_ratio=0.015,
        # synthetic short record: the sampling guard is not the subject here
        check_sampling=False,
    )
    response = build_building_dynamic_response(_floor_source(cf_x, cf_y, cm_z), cfg)
    acc = build_point_accelerations(
        response, ComfortConfig(cm_positions=df_floors[["XR", "YR"]].to_numpy(), point=POINT)
    )

    for field in ("acc_x", "acc_y"):
        nrms = _nrms(acc.fields.read(field), golden(f"cf_{field}"))
        assert nrms < 2e-2, f"{field} drifted from the legacy solve: nrms={nrms:.2e}"


def test_acceleration_magnitude_and_peak_reduction():
    cf_x, cf_y, cm_z, df_floors, df_modes, shapes = _synthetic()
    cfg = BuildingDynamicConfig(
        mode_shapes=np.stack(
            [np.column_stack([s["DX"], s["DY"], s["RZ"]]) for s in shapes], axis=1
        ),
        floor_points=np.zeros((N_FLOORS, 3)),
        cm_positions=df_floors[["XR", "YR"]].to_numpy(),
        floors_mass=df_floors["M"].to_numpy(),
        floors_radius=df_floors["R"].to_numpy(),
        natural_frequencies=df_modes["wp"].to_numpy(),
        damping_ratio=0.02,
        # synthetic short record: the sampling guard is not the subject here
        check_sampling=False,
    )
    response = build_building_dynamic_response(_floor_source(cf_x, cf_y, cm_z), cfg)
    acc = build_point_accelerations(
        response, ComfortConfig(cm_positions=df_floors[["XR", "YR"]].to_numpy(), point=POINT)
    )

    acc_mag = np.asarray(acc.fields.read("acc_mag"))
    acc_x = np.asarray(acc.fields.read("acc_x"))
    acc_y = np.asarray(acc.fields.read("acc_y"))
    assert acc_mag.shape == (N_FLOORS, N_T)
    np.testing.assert_allclose(acc_mag, np.hypot(acc_x, acc_y))
    assert np.all(acc_mag >= 0)

    # Peak reduction via the shared extreme_value op (peak_factor).
    peak = extreme_value(
        acc,
        ExtremeValueParams(
            method="peak_factor", extreme_type="max", field="acc_mag", peak_factor=3.0
        ),
    )
    reduced = np.asarray(peak.fields.read("peak_factor_max"))
    assert reduced.shape == (N_FLOORS,)
    assert np.isfinite(reduced).all()
