"""Generate frozen legacy-parity golden values for the dynamics tests.

Run ONCE while ``cfdmod.hfpi`` still exists:

    uv run python fixtures/tests/dynamics/_generate_goldens.py

It reuses the exact input builders from the test modules (single source of
truth, no drift) and captures the legacy ``cfdmod.hfpi`` outputs into
``legacy_goldens.npz``. The dynamics tests load that archive instead of
importing the legacy module, so they keep guarding the v3 code after
``cfdmod/hfpi/`` is deleted.
"""

from __future__ import annotations

import pathlib
import tempfile

import numpy as np
import pandas as pd

HERE = pathlib.Path(__file__).parent


def _building_dynamic():
    from cfdmod.hfpi import dynamic as ld
    from cfdmod.hfpi import static as ls

    from tests.core.recipes.test_building_dynamic import DT, N_FLOORS, N_T, _synthetic_inputs

    cf_x, cf_y, cm_z, df_floors, df_modes, shapes = _synthetic_inputs()

    def force_df(arr):
        df = pd.DataFrame({str(f): arr[f] for f in range(N_FLOORS)})
        df["time_normalized"] = np.arange(N_T) * DT
        return df

    forces = ls.StaticForcesData(cf_x=force_df(cf_x), cf_y=force_df(cf_y), cm_z=force_df(cm_z))
    u_h = (1.0 / 0.613) ** 0.5
    dim = ls.DimensionalData(U_H=u_h, height=1.0, base=1.0, integral_scale_multiplier=u_h)
    struct = ld.HFPIStructuralData(
        df_modes=df_modes, df_floors=df_floors, df_modal_shapes=shapes, active_modes=[0, 1]
    )
    res = ld.solve_hfpi(structural_data=struct, dim_data=dim, forces=forces, xi=0.015)
    return {
        "bd_disp_x": res.displacement["x"].T,
        "bd_disp_y": res.displacement["y"].T,
        "bd_rot_z": res.displacement["z"].T,
        "bd_feq_x": res.forces_static_eq["x"].T,
        "bd_feq_y": res.forces_static_eq["y"].T,
        "bd_meq_z": res.forces_static_eq["z"].T,
    }


def _forces_scaling():
    from cfdmod.hfpi import static as ls

    from tests.dynamics.test_forces import N_FLOORS, _write_forces

    with tempfile.TemporaryDirectory() as d:
        paths, _ = _write_forces(pathlib.Path(d))
        legacy = ls.StaticForcesData.build(paths["cf_x"], paths["cf_y"], paths["cm_z"])
        dim = ls.DimensionalData(U_H=22.0, height=90.0, base=35.0, integral_scale_multiplier=1.3)
        scaled = legacy.get_scaled_forces(dim)
        scaled.fill_missing_floors(N_FLOORS)
        dct = scaled.get_as_dct()
    return {"fs_x": dct["x"].T, "fs_y": dct["y"].T, "fs_z": dct["z"].T}


def _mass_normalization():
    from cfdmod.hfpi.dynamic import normalize_mode_shapes

    n_floors = 4
    rng = np.random.default_rng(0)
    mass = np.linspace(100, 130, n_floors)
    radius = (np.linspace(800, 1100, n_floors) / mass) ** 0.5
    df_floors = pd.DataFrame({"M": mass, "R": radius})
    df_phi = pd.DataFrame(
        {
            "DX": rng.normal(size=n_floors),
            "DY": rng.normal(size=n_floors),
            "RZ": rng.normal(size=n_floors) * 0.01,
        }
    )
    normalize_mode_shapes(df_floors, df_phi)  # in place
    return {"mn_norm": np.column_stack([df_phi["DX"], df_phi["DY"], df_phi["RZ"]])}


def _multiplier():
    from cfdmod.hfpi import dynamic as ld
    from cfdmod.hfpi import static as ls

    from tests.dynamics.test_cases import DT, N_FLOORS, N_T, _raw_inputs

    cf_x, cf_y, cm_z, df_floors, df_modes, shapes = _raw_inputs()
    mm, fm, xi = 1.4, 1.2, 0.015
    df_floors_mod = df_floors.copy()
    df_floors_mod["M"] = df_floors_mod["M"] * mm
    df_modes_mod = df_modes.copy()
    df_modes_mod["frequency"] = df_modes_mod["frequency"] / (mm**0.5) * fm
    df_modes_mod["wp"] = df_modes_mod["frequency"] * 2 * np.pi
    df_modes_mod["period"] = 1 / df_modes_mod["frequency"]

    def force_df(arr):
        df = pd.DataFrame({str(f): arr[f] for f in range(N_FLOORS)})
        df["time_normalized"] = np.arange(N_T) * DT
        return df

    forces = ls.StaticForcesData(cf_x=force_df(cf_x), cf_y=force_df(cf_y), cm_z=force_df(cm_z))
    u_h = (1.0 / 0.613) ** 0.5
    dim = ls.DimensionalData(U_H=u_h, height=1.0, base=1.0, integral_scale_multiplier=u_h)
    struct = ld.HFPIStructuralData(
        df_modes=df_modes_mod,
        df_floors=df_floors_mod,
        df_modal_shapes=[s.copy() for s in shapes],
        active_modes=[0, 1],
    )
    res = ld.solve_hfpi(structural_data=struct, dim_data=dim, forces=forces, xi=xi)
    return {
        "mul_disp_x": res.displacement["x"].T,
        "mul_feq_x": res.forces_static_eq["x"].T,
        "mul_meq_z": res.forces_static_eq["z"].T,
    }


def _comfort():
    from cfdmod.hfpi import dynamic as ld
    from cfdmod.hfpi import static as ls

    from tests.core.recipes.test_comfort import DT, N_FLOORS, N_T, POINT, _synthetic

    cf_x, cf_y, cm_z, df_floors, df_modes, shapes = _synthetic()

    def force_df(arr):
        df = pd.DataFrame({str(f): arr[f] for f in range(N_FLOORS)})
        df["time_normalized"] = np.arange(N_T) * DT
        return df

    forces = ls.StaticForcesData(cf_x=force_df(cf_x), cf_y=force_df(cf_y), cm_z=force_df(cm_z))
    u_h = (1.0 / 0.613) ** 0.5
    dim = ls.DimensionalData(U_H=u_h, height=1.0, base=1.0, integral_scale_multiplier=u_h)
    struct = ld.HFPIStructuralData(
        df_modes=df_modes,
        df_floors=df_floors,
        df_modal_shapes=[s.copy() for s in shapes],
        active_modes=[0, 1],
    )
    res = ld.solve_hfpi(structural_data=struct, dim_data=dim, forces=forces, xi=0.015)
    acc = res.get_point_acceleration(df_floors[["XR", "YR"]], POINT)
    return {"cf_acc_x": acc["x"].T, "cf_acc_y": acc["y"].T}


def main():
    goldens: dict[str, np.ndarray] = {}
    for fn in (_building_dynamic, _forces_scaling, _mass_normalization, _multiplier, _comfort):
        goldens.update(fn())
    out = HERE / "legacy_goldens.npz"
    np.savez_compressed(out, **goldens)
    print(f"wrote {out} with keys: {sorted(goldens)}")


if __name__ == "__main__":
    main()
