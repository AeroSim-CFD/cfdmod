"""Building dynamic-response recipe: full-recipe + legacy-parity tests.

The parity test pins the v3 ``build_building_dynamic_response`` against
frozen legacy ``solve_hfpi`` outputs (``fixtures/tests/dynamics/legacy_goldens.npz``,
generated while ``cfdmod.hfpi`` still existed), so it keeps guarding the v3
code after the legacy module was removed.
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
    """A very stiff single mode -> quasi-static static-equivalent force.

    For a mode much stiffer than the forcing, ``q -> Q / wp^2`` (quasi-static),
    so ``feq_x = M * wp^2 * DX * q -> M * DX * Q`` with the ``wp^2`` cancelling.
    We assert that limit directly, independent of ``wp``.
    """
    from cfdmod.core.ops.data_source_create.generalized_building_load import (
        GeneralizedBuildingLoadParams,
        generalized_building_load,
    )

    cf_x, cf_y, cm_z, df_floors, _, modal_shapes = _synthetic_inputs()
    phi = np.stack(
        [np.column_stack([s["DX"], s["DY"], s["RZ"]]) for s in modal_shapes[:1]], axis=1
    )  # single mode
    # 20 Hz vs the ~0.3-0.7 Hz forcing: (w/wp)^2 ~ 1e-3 so the response is
    # quasi-static, while still well below the output Nyquist (10 Hz at dt=0.05).
    wp = np.array([2 * np.pi * 20.0])
    cm_positions = df_floors[["XR", "YR"]].to_numpy()
    mass = df_floors["M"].to_numpy()
    load = _floor_source(cf_x, cf_y, cm_z)

    cfg = BuildingDynamicConfig(
        mode_shapes=phi,
        floor_points=np.zeros((N_FLOORS, 3)),
        cm_positions=cm_positions,
        floors_mass=mass,
        floors_radius=df_floors["R"].to_numpy(),
        natural_frequencies=wp,
        damping_ratio=0.02,
    )
    out = build_building_dynamic_response(load, cfg)

    # Generalized load Q(t) for the single mode.
    modes = generalized_building_load(
        load, GeneralizedBuildingLoadParams(mode_shapes=phi, cm_positions=cm_positions, out="q")
    )
    q_load = np.asarray(modes.fields.read("q"))[0]  # (n_t,)

    # Quasi-static limit: feq_x[floor, t] -> M[floor] * DX[floor] * Q(t) (the wp^2
    # cancels). Compare a tail window (after the startup transient) via a
    # normalized RMS error, which is robust to Q's zero crossings.
    expected = mass[:, None] * phi[:, 0, 0][:, None] * q_load[None, :]
    feq = np.asarray(out.fields.read("feq_x"))
    assert np.isfinite(feq).all()

    tail = slice(60, None)
    nrms = np.sqrt(np.mean((feq[:, tail] - expected[:, tail]) ** 2)) / np.sqrt(
        np.mean(expected[:, tail] ** 2)
    )
    assert nrms < 1e-2, f"quasi-static normalized RMS error too large: {nrms:.2e}"


def test_legacy_parity_displacement_and_static_equivalent_forces():
    """v3 recipe matches frozen legacy solve_hfpi outputs (characterization)."""
    from cfdmod.dynamics.structural import mass_normalize_mode_shapes
    from tests.dynamics._goldens import golden

    cf_x, cf_y, cm_z, df_floors, df_modes, modal_shapes = _synthetic_inputs()

    # Mass-normalize the raw shapes the way solve_hfpi does internally
    # (mass_normalize_mode_shapes is itself pinned against the legacy
    # normalize_mode_shapes in test_structural).
    phi_raw = np.stack(
        [np.column_stack([s["DX"], s["DY"], s["RZ"]]) for s in modal_shapes], axis=1
    )
    phi = mass_normalize_mode_shapes(phi_raw, df_floors["M"].to_numpy(), df_floors["R"].to_numpy())

    cfg = BuildingDynamicConfig(
        mode_shapes=phi,
        floor_points=np.column_stack([np.zeros(N_FLOORS), np.zeros(N_FLOORS), df_floors["Z"]]),
        cm_positions=df_floors[["XR", "YR"]].to_numpy(),
        floors_mass=df_floors["M"].to_numpy(),
        floors_radius=df_floors["R"].to_numpy(),
        natural_frequencies=df_modes["wp"].to_numpy(),
        damping_ratio=0.015,
    )
    out = build_building_dynamic_response(_floor_source(cf_x, cf_y, cm_z), cfg)

    np.testing.assert_allclose(
        out.fields.read("disp_x"), golden("bd_disp_x"), rtol=1e-6, atol=1e-9
    )
    np.testing.assert_allclose(
        out.fields.read("disp_y"), golden("bd_disp_y"), rtol=1e-6, atol=1e-9
    )
    np.testing.assert_allclose(out.fields.read("rot_z"), golden("bd_rot_z"), rtol=1e-6, atol=1e-9)
    np.testing.assert_allclose(out.fields.read("feq_x"), golden("bd_feq_x"), rtol=1e-6, atol=1e-9)
    np.testing.assert_allclose(out.fields.read("feq_y"), golden("bd_feq_y"), rtol=1e-6, atol=1e-9)
    np.testing.assert_allclose(out.fields.read("meq_z"), golden("bd_meq_z"), rtol=1e-6, atol=1e-9)
