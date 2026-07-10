"""Tests for the building structural-data CSV ingest."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from cfdmod.dynamics import (
    BuildingStructuralData,
    mass_normalize_mode_shapes,
    read_floors_csv,
    read_modes_csv,
)

N_FLOORS = 4
N_MODES = 3


@pytest.fixture
def csv_dir(tmp_path):
    heights = np.arange(1, N_FLOORS + 1) * 3.0
    pd.DataFrame(
        {
            "Z": heights,
            "M": np.linspace(100, 130, N_FLOORS),
            "I": np.linspace(800, 1100, N_FLOORS),
            "XR": np.full(N_FLOORS, 0.4),
            "YR": np.full(N_FLOORS, 0.25),
        }
    ).to_csv(tmp_path / "floors.csv", index=False)

    pd.DataFrame({"mode": [1, 2, 3], "period": [1.2, 0.6, 0.3]}).to_csv(
        tmp_path / "modes.csv", index=False
    )

    shape_paths = []
    for m in range(N_MODES):
        p = tmp_path / f"phi_{m}.csv"
        pd.DataFrame(
            {
                "DX": np.linspace(0.1, 1.0, N_FLOORS) * (1 + m),
                "DY": np.linspace(0.05, 0.5, N_FLOORS),
                "RZ": np.linspace(0.01, 0.04, N_FLOORS),
            }
        ).to_csv(p, index=False)
        shape_paths.append(p)

    return tmp_path, shape_paths


def test_read_modes_adds_frequency_and_angular_frequency(csv_dir):
    tmp_path, _ = csv_dir
    df = read_modes_csv(tmp_path / "modes.csv")
    np.testing.assert_allclose(df["frequency"], 1 / df["period"])
    np.testing.assert_allclose(df["wp"], 2 * np.pi * df["frequency"])


def test_read_floors_derives_radius_of_gyration(csv_dir):
    tmp_path, _ = csv_dir
    df = read_floors_csv(tmp_path / "floors.csv")
    np.testing.assert_allclose(df["R"], (df["I"] / df["M"]) ** 0.5)


def test_from_csvs_assembles_normalized_recipe_inputs(csv_dir):
    tmp_path, shape_paths = csv_dir
    data = BuildingStructuralData.from_csvs(
        tmp_path / "modes.csv", tmp_path / "floors.csv", shape_paths
    )
    assert data.n_floors == N_FLOORS
    assert data.n_modes == N_MODES
    assert np.asarray(data.mode_shapes).shape == (N_FLOORS, N_MODES, 3)
    assert np.asarray(data.natural_frequencies).shape == (N_MODES,)

    # Each mode has unit generalized mass after normalization.
    phi = np.asarray(data.mode_shapes)
    m = np.asarray(data.floors_mass)[:, None]
    r = np.asarray(data.floors_radius)[:, None]
    m_gen = (m * (phi[:, :, 0] ** 2 + phi[:, :, 1] ** 2 + (r * phi[:, :, 2]) ** 2)).sum(axis=0)
    np.testing.assert_allclose(m_gen, np.ones(N_MODES), rtol=1e-9)


def test_active_modes_selection(csv_dir):
    tmp_path, shape_paths = csv_dir
    data = BuildingStructuralData.from_csvs(
        tmp_path / "modes.csv", tmp_path / "floors.csv", shape_paths, active_modes=[1, 3]
    )
    assert data.n_modes == 2


def test_mass_normalization_matches_legacy():
    """mass_normalize_mode_shapes reproduces legacy hfpi normalize_mode_shapes."""
    from cfdmod.hfpi.dynamic import normalize_mode_shapes

    rng = np.random.default_rng(0)
    mass = np.linspace(100, 130, N_FLOORS)
    radius = (np.linspace(800, 1100, N_FLOORS) / mass) ** 0.5

    df_floors = pd.DataFrame({"M": mass, "R": radius})
    df_phi = pd.DataFrame(
        {
            "DX": rng.normal(size=N_FLOORS),
            "DY": rng.normal(size=N_FLOORS),
            "RZ": rng.normal(size=N_FLOORS) * 0.01,
        }
    )
    phi = np.column_stack([df_phi["DX"], df_phi["DY"], df_phi["RZ"]])[:, None, :]

    got = mass_normalize_mode_shapes(phi, mass, radius)[:, 0, :]

    normalize_mode_shapes(df_floors, df_phi)  # mutates df_phi in place
    expected = np.column_stack([df_phi["DX"], df_phi["DY"], df_phi["RZ"]])

    np.testing.assert_allclose(got, expected, rtol=1e-12)
