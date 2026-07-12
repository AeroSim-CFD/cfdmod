"""Tests for the TQS PORTELS reader (compact anonymized fixture)."""

from __future__ import annotations

import pathlib

import numpy as np

from cfdmod.adapters.memory import MemoryFieldStore
from cfdmod.core import ElementMeta, PointsDataSource, TimeAxis, Topology
from cfdmod.core.recipes import build_building_dynamic_response
from cfdmod.dynamics import BuildingStructuralData, read_tqs_portels
from cfdmod.dynamics.imports._csv_out import write_structural_csvs

FIX = pathlib.Path(__file__).resolve().parents[2] / "fixtures" / "tests" / "dynamics" / "imports"
TQS = FIX / "tqs"


def test_reads_shapes_periods_and_geometry():
    sd = read_tqs_portels(TQS)
    assert sd.n_floors == 3
    assert sd.n_modes == 2
    np.testing.assert_allclose(np.asarray(sd.floor_points)[:, 2], [3.0, 6.0, 9.0])
    # periods 1.0 / 0.4 s -> 1.0 / 2.5 Hz
    np.testing.assert_allclose(np.asarray(sd.natural_frequencies) / (2 * np.pi), [1.0, 2.5])
    # CoM from unequal corner masses: (0, 0.5).
    np.testing.assert_allclose(np.asarray(sd.cm_positions)[0], [0.0, 0.5], atol=1e-9)


def test_mode_shapes_are_mass_normalized():
    sd = read_tqs_portels(TQS)
    phi = np.asarray(sd.mode_shapes)
    m = np.asarray(sd.floors_mass)[:, None]
    r = np.asarray(sd.floors_radius)[:, None]
    m_gen = (m * (phi[:, :, 0] ** 2 + phi[:, :, 1] ** 2 + (r * phi[:, :, 2]) ** 2)).sum(axis=0)
    np.testing.assert_allclose(m_gen, np.ones(2), rtol=1e-9)


def test_active_modes_selection():
    sd = read_tqs_portels(TQS, active_modes=[1])
    assert sd.n_modes == 1
    np.testing.assert_allclose(np.asarray(sd.natural_frequencies) / (2 * np.pi), [1.0])


def test_csv_round_trip_is_idempotent(tmp_path):
    sd = read_tqs_portels(TQS)
    paths = write_structural_csvs(sd, tmp_path)
    phis = sorted(p for p in paths if p.name.startswith("phi"))
    rt = BuildingStructuralData.from_csvs(tmp_path / "modes.csv", tmp_path / "floors.csv", phis)
    np.testing.assert_allclose(np.asarray(rt.mode_shapes), np.asarray(sd.mode_shapes), rtol=1e-9)
    np.testing.assert_allclose(rt.natural_frequencies, sd.natural_frequencies, rtol=1e-9)
    np.testing.assert_allclose(rt.floors_radius, sd.floors_radius, rtol=1e-9)


def test_uses_pisos_floor_levels_and_names():
    # The fixture ships a PORTELSSE_PISOS.TXT with 3 floors at 3/6/9 m; the
    # recovered elevations must match it (not the raw node Z clustering), and
    # the storey names are kept as metadata labels.
    sd = read_tqs_portels(TQS)
    np.testing.assert_allclose(np.asarray(sd.floor_points)[:, 2], [3.0, 6.0, 9.0])
    assert sd.floor_labels == ["Pav 1", "Pav 2", "Pav 3"]


def test_old_prefix_without_pisos(tmp_path):
    # Older exports use the PORTELS_ prefix and ship no PISOS table; the reader
    # must still find the files (suffix match) and recover floors by clustering.
    for p in TQS.glob("PORTELSSE_*.TXT"):
        if p.name.endswith("_PISOS.TXT"):
            continue
        (tmp_path / p.name.replace("PORTELSSE_", "PORTELS_")).write_bytes(p.read_bytes())
    sd = read_tqs_portels(tmp_path)
    assert sd.n_floors == 3
    np.testing.assert_allclose(np.asarray(sd.natural_frequencies) / (2 * np.pi), [1.0, 2.5])


def test_feeds_building_dynamic_recipe():
    sd = read_tqs_portels(TQS)
    cfg = sd.to_config(damping_ratio=0.02)
    n_floors, n_t, dt = sd.n_floors, 128, 0.05
    rng = np.random.default_rng(0)
    t = np.arange(n_t) * dt
    load = PointsDataSource(
        time=TimeAxis(initial_time=0.0, timestep_size=dt, n_timesteps=n_t),
        topology=Topology.points(np.asarray(sd.floor_points)),
        elements=ElementMeta(position=np.asarray(sd.floor_points)),
        fields=MemoryFieldStore(
            {
                "cf_x": np.sin(2 * np.pi * 0.3 * t) + 0.01 * rng.standard_normal((n_floors, n_t)),
                "cf_y": 0.5 * np.cos(2 * np.pi * 0.4 * t) * np.ones((n_floors, n_t)),
                "cm_z": 0.1 * np.sin(2 * np.pi * 0.5 * t) * np.ones((n_floors, n_t)),
            }
        ),
    )
    resp = build_building_dynamic_response(load, cfg)
    for name in ("disp_x", "disp_y", "rot_z", "feq_x", "feq_y", "meq_z"):
        arr = np.asarray(resp.fields.read(name))
        assert arr.shape == (n_floors, n_t)
        assert np.all(np.isfinite(arr))
