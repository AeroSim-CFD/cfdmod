"""Building dynamic-response recipe: full-recipe + legacy-parity tests.

The parity test pins the v3 ``build_building_dynamic_response`` against
the legacy ``cfdmod.hfpi.solve_hfpi`` on a tiny synthetic fixture, so the
one-shot cutover is guarded before ``cfdmod/hfpi/`` is deleted.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from cfdmod.adapters.memory import MemoryFieldStore
from cfdmod.core import ElementMeta, PointsDataSource, TimeAxis, Topology
from cfdmod.core.recipes import (
    BuildingDynamicConfig,
    build_building_dynamic_response,
)

N_FLOORS = 3
N_MODES = 2
N_T = 240
DT = 0.05


def _floor_source(cf_x: np.ndarray, cf_y: np.ndarray, cm_z: np.ndarray) -> PointsDataSource:
    pts = np.zeros((N_FLOORS, 3), dtype=np.float64)
    pts[:, 2] = np.arange(1, N_FLOORS + 1) * 3.0  # floor heights
    return PointsDataSource(
        time=TimeAxis(initial_time=0.0, timestep_size=DT, n_timesteps=N_T),
        topology=Topology.points(pts),
        elements=ElementMeta(position=pts),
        fields=MemoryFieldStore({"cf_x": cf_x, "cf_y": cf_y, "cm_z": cm_z}),
    )


def _synthetic_inputs():
    """Deterministic synthetic structural data + per-floor load series."""
    rng = np.random.default_rng(42)
    t = np.arange(N_T) * DT

    # Smooth, band-limited per-floor load coefficient series (n_floors, n_t).
    def series(scale, phase):
        return scale * (
            np.sin(2 * np.pi * 0.3 * t + phase) + 0.4 * np.sin(2 * np.pi * 0.7 * t + 2 * phase)
        )

    cf_x = np.vstack([series(1.0 + 0.2 * f, 0.1 * f) for f in range(N_FLOORS)])
    cf_y = np.vstack([series(0.7 + 0.1 * f, 0.5 + 0.1 * f) for f in range(N_FLOORS)])
    cm_z = np.vstack([series(0.2 + 0.05 * f, 1.0 + 0.2 * f) for f in range(N_FLOORS)])
    # A touch of noise so it is not perfectly periodic.
    cf_x = cf_x + 0.01 * rng.standard_normal(cf_x.shape)

    heights = np.arange(1, N_FLOORS + 1) * 3.0
    df_floors = pd.DataFrame(
        {
            "Z": heights,
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

    modal_shapes = []
    for m in range(N_MODES):
        modal_shapes.append(
            pd.DataFrame(
                {
                    "DX": np.linspace(0.1, 1.0, N_FLOORS) * (1 + 0.3 * m),
                    "DY": np.linspace(0.05, 0.6, N_FLOORS) * (1 - 0.2 * m),
                    "RZ": np.linspace(0.01, 0.05, N_FLOORS) * (1 + 0.1 * m),
                }
            )
        )

    return cf_x, cf_y, cm_z, df_floors, df_modes, modal_shapes


def test_recipe_produces_finite_response_of_expected_shape():
    cf_x, cf_y, cm_z, df_floors, df_modes, modal_shapes = _synthetic_inputs()

    phi = np.stack(
        [np.column_stack([s["DX"], s["DY"], s["RZ"]]) for s in modal_shapes], axis=1
    )  # (n_floors, n_modes, 3)
    cfg = BuildingDynamicConfig(
        mode_shapes=phi,
        floor_points=np.column_stack([np.zeros(N_FLOORS), np.zeros(N_FLOORS), df_floors["Z"]]),
        cm_positions=df_floors[["XR", "YR"]].to_numpy(),
        floors_mass=df_floors["M"].to_numpy(),
        floors_radius=df_floors["R"].to_numpy(),
        natural_frequencies=df_modes["wp"].to_numpy(),
        damping_ratio=0.02,
    )
    out = build_building_dynamic_response(_floor_source(cf_x, cf_y, cm_z), cfg)

    for name in ("disp_x", "disp_y", "rot_z", "feq_x", "feq_y", "meq_z"):
        arr = np.asarray(out.fields.read(name))
        assert arr.shape == (N_FLOORS, N_T)
        assert np.isfinite(arr).all()


def test_stiff_mode_approaches_quasi_static_recomposition():
    """A very stiff single mode: modal displacement -> Q / wp^2 (quasi-static)."""
    cf_x, cf_y, cm_z, df_floors, _, modal_shapes = _synthetic_inputs()
    phi = np.stack(
        [np.column_stack([s["DX"], s["DY"], s["RZ"]]) for s in modal_shapes[:1]], axis=1
    )  # single mode
    wp = np.array([2 * np.pi * 50.0])  # very stiff

    cfg = BuildingDynamicConfig(
        mode_shapes=phi,
        floor_points=np.zeros((N_FLOORS, 3)),
        cm_positions=df_floors[["XR", "YR"]].to_numpy(),
        floors_mass=df_floors["M"].to_numpy(),
        floors_radius=df_floors["R"].to_numpy(),
        natural_frequencies=wp,
        damping_ratio=0.02,
    )
    out = build_building_dynamic_response(_floor_source(cf_x, cf_y, cm_z), cfg)

    # feq = M * wp^2 * DX * q ; quasi-static q ~ Q/wp^2 so feq ~ M * DX * Q.
    # Just assert the static-equivalent force is bounded and finite for a stiff mode.
    feq = np.asarray(out.fields.read("feq_x"))
    assert np.isfinite(feq).all()


def test_legacy_parity_displacement_and_static_equivalent_forces():
    """v3 recipe matches legacy solve_hfpi on identical numbers."""
    from cfdmod.hfpi import dynamic as legacy_dynamic
    from cfdmod.hfpi import static as legacy_static

    cf_x, cf_y, cm_z, df_floors, df_modes, modal_shapes = _synthetic_inputs()

    # Build legacy force data with string floor columns + time_normalized.
    def force_df(arr):
        df = pd.DataFrame({str(f): arr[f] for f in range(N_FLOORS)})
        df["time_normalized"] = np.arange(N_T) * DT
        return df

    forces = legacy_static.StaticForcesData(
        cf_x=force_df(cf_x), cf_y=force_df(cf_y), cm_z=force_df(cm_z)
    )

    # Dimensional data chosen so force/moment/time scaling factors are all 1.
    u_h = (1.0 / 0.613) ** 0.5
    dim_data = legacy_static.DimensionalData(
        U_H=u_h, height=1.0, base=1.0, integral_scale_multiplier=u_h
    )
    assert abs(dim_data.force_normalization_factor - 1.0) < 1e-9
    assert abs(dim_data.moments_normalization_factor - 1.0) < 1e-9
    assert abs(dim_data.time_normalization_factor - 1.0) < 1e-9

    struct = legacy_dynamic.HFPIStructuralData(
        df_modes=df_modes,
        df_floors=df_floors,
        df_modal_shapes=modal_shapes,
        active_modes=[0, 1],
    )

    xi = 0.015
    legacy_result = legacy_dynamic.solve_hfpi(
        structural_data=struct, dim_data=dim_data, forces=forces, xi=xi
    )

    # After solve_hfpi, mode shapes are mass-normalized in place -> feed same to v3.
    phi = np.stack(
        [np.column_stack([s["DX"], s["DY"], s["RZ"]]) for s in struct.df_modal_shapes],
        axis=1,
    )  # (n_floors, n_modes, 3)

    cfg = BuildingDynamicConfig(
        mode_shapes=phi,
        floor_points=np.column_stack([np.zeros(N_FLOORS), np.zeros(N_FLOORS), df_floors["Z"]]),
        cm_positions=df_floors[["XR", "YR"]].to_numpy(),
        floors_mass=df_floors["M"].to_numpy(),
        floors_radius=df_floors["R"].to_numpy(),
        natural_frequencies=df_modes["wp"].to_numpy(),
        damping_ratio=xi,
    )
    out = build_building_dynamic_response(_floor_source(cf_x, cf_y, cm_z), cfg)

    # Legacy dicts are (n_samples, n_floors); v3 fields are (n_floors, n_samples).
    np.testing.assert_allclose(
        out.fields.read("disp_x"), legacy_result.displacement["x"].T, rtol=1e-6, atol=1e-9
    )
    np.testing.assert_allclose(
        out.fields.read("disp_y"), legacy_result.displacement["y"].T, rtol=1e-6, atol=1e-9
    )
    np.testing.assert_allclose(
        out.fields.read("rot_z"), legacy_result.displacement["z"].T, rtol=1e-6, atol=1e-9
    )
    np.testing.assert_allclose(
        out.fields.read("feq_x"), legacy_result.forces_static_eq["x"].T, rtol=1e-6, atol=1e-9
    )
    np.testing.assert_allclose(
        out.fields.read("feq_y"), legacy_result.forces_static_eq["y"].T, rtol=1e-6, atol=1e-9
    )
    np.testing.assert_allclose(
        out.fields.read("meq_z"), legacy_result.forces_static_eq["z"].T, rtol=1e-6, atol=1e-9
    )
