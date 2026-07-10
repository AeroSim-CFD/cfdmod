"""Tests for the building force-coefficient ingest + end-to-end from disk."""

from __future__ import annotations

import numpy as np
import pandas as pd

from cfdmod.dynamics import (
    DimensionalData,
    build_floor_load_source,
    read_force_h5,
    write_force_h5,
)

N_FLOORS = 3
N_T = 120


def _force_df(rng, scale):
    df = pd.DataFrame({str(f): scale * rng.standard_normal(N_T) for f in range(N_FLOORS)})
    df["time_normalized"] = np.arange(N_T) * 0.5
    return df


def _write_forces(tmp_path):
    rng = np.random.default_rng(7)
    paths = {}
    dfs = {}
    for name, scale in (("cf_x", 1.0), ("cf_y", 0.8), ("cm_z", 0.3)):
        df = _force_df(rng, scale)
        p = tmp_path / f"{name}.h5"
        write_force_h5(df, p)
        paths[name] = p
        dfs[name] = df
    return paths, dfs


def test_h5_round_trip(tmp_path):
    rng = np.random.default_rng(1)
    df = _force_df(rng, 1.0)
    p = tmp_path / "f.h5"
    write_force_h5(df, p)
    back = read_force_h5(p)
    pd.testing.assert_frame_equal(df, back, check_dtype=False)


def test_dimensional_data_factors():
    dim = DimensionalData(U_H=30.0, height=100.0, base=40.0, integral_scale_multiplier=2.0)
    assert dim.dynamic_pressure == 0.613 * 30.0**2
    assert dim.CST == 40.0 / 30.0
    assert dim.time_normalization_factor == (40.0 / 30.0) * 2.0
    assert dim.force_normalization_factor == 40.0 * 100.0 * dim.dynamic_pressure
    assert dim.moments_normalization_factor == 40.0 * 40.0 * 100.0 * dim.dynamic_pressure


def test_build_floor_load_source_shapes_and_scaling(tmp_path):
    paths, dfs = _write_forces(tmp_path)
    dim = DimensionalData(U_H=25.0, height=80.0, base=30.0, integral_scale_multiplier=1.5)

    src = build_floor_load_source(
        paths["cf_x"], paths["cf_y"], paths["cm_z"], dim, n_floors=N_FLOORS
    )
    cf_x = np.asarray(src.fields.read("cf_x"))
    assert cf_x.shape == (N_FLOORS, N_T)

    # Column "0" scaled by the force factor.
    expected0 = dfs["cf_x"]["0"].to_numpy() * dim.force_normalization_factor
    np.testing.assert_allclose(cf_x[0], expected0, rtol=1e-12)
    # dt scaled by the time factor.
    np.testing.assert_allclose(
        src.time.timestep_size, 0.5 * dim.time_normalization_factor, rtol=1e-12
    )


def test_missing_floor_filled_with_zeros(tmp_path):
    rng = np.random.default_rng(3)
    df = pd.DataFrame({"0": rng.standard_normal(N_T), "2": rng.standard_normal(N_T)})
    df["time_normalized"] = np.arange(N_T) * 0.5
    for name in ("cf_x", "cf_y", "cm_z"):
        write_force_h5(df, tmp_path / f"{name}.h5")
    dim = DimensionalData(U_H=25.0, height=80.0, base=30.0, integral_scale_multiplier=1.0)

    src = build_floor_load_source(
        tmp_path / "cf_x.h5", tmp_path / "cf_y.h5", tmp_path / "cm_z.h5", dim, n_floors=3
    )
    cf_x = np.asarray(src.fields.read("cf_x"))
    # Floor 1 was absent -> zeros.
    np.testing.assert_array_equal(cf_x[1], np.zeros(N_T))


def test_parity_with_legacy_scaling(tmp_path):
    """Scaled floor matrices match the legacy StaticForcesData path."""
    from cfdmod.hfpi import static as legacy_static

    paths, _ = _write_forces(tmp_path)
    dim = DimensionalData(U_H=22.0, height=90.0, base=35.0, integral_scale_multiplier=1.3)

    src = build_floor_load_source(
        paths["cf_x"], paths["cf_y"], paths["cm_z"], dim, n_floors=N_FLOORS
    )

    legacy_forces = legacy_static.StaticForcesData.build(
        paths["cf_x"], paths["cf_y"], paths["cm_z"]
    )
    legacy_dim = legacy_static.DimensionalData(
        U_H=22.0, height=90.0, base=35.0, integral_scale_multiplier=1.3
    )
    legacy_scaled = legacy_forces.get_scaled_forces(legacy_dim)
    legacy_scaled.fill_missing_floors(N_FLOORS)
    legacy_dct = legacy_scaled.get_as_dct()  # (n_samples, n_floors) per axis

    np.testing.assert_allclose(src.fields.read("cf_x"), legacy_dct["x"].T, rtol=1e-10)
    np.testing.assert_allclose(src.fields.read("cf_y"), legacy_dct["y"].T, rtol=1e-10)
    np.testing.assert_allclose(src.fields.read("cm_z"), legacy_dct["z"].T, rtol=1e-10)


def test_end_to_end_disk_to_recipe(tmp_path):
    """Force H5 + structural arrays -> full building dynamic response."""
    from cfdmod.core.recipes import (
        BuildingDynamicConfig,
        build_building_dynamic_response,
    )

    paths, _ = _write_forces(tmp_path)
    dim = DimensionalData(U_H=25.0, height=80.0, base=30.0, integral_scale_multiplier=1.0)
    src = build_floor_load_source(
        paths["cf_x"], paths["cf_y"], paths["cm_z"], dim, n_floors=N_FLOORS
    )

    n_modes = 2
    rng = np.random.default_rng(0)
    phi = rng.standard_normal((N_FLOORS, n_modes, 3)) * 0.1
    cfg = BuildingDynamicConfig(
        mode_shapes=phi,
        floor_points=np.zeros((N_FLOORS, 3)),
        cm_positions=np.tile([0.4, 0.2], (N_FLOORS, 1)),
        floors_mass=np.full(N_FLOORS, 120.0),
        floors_radius=np.full(N_FLOORS, 2.7),
        natural_frequencies=np.array([2 * np.pi * 1.0, 2 * np.pi * 2.5]),
        damping_ratio=0.02,
    )
    out = build_building_dynamic_response(src, cfg)
    for name in ("disp_x", "disp_y", "rot_z", "feq_x", "feq_y", "meq_z"):
        arr = np.asarray(out.fields.read(name))
        assert arr.shape == (N_FLOORS, N_T)
        assert np.isfinite(arr).all()
