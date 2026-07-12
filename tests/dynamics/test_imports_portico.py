"""Tests for the TQS Portico per-floor reader (PAVIMENTO variant)."""

from __future__ import annotations

import pathlib

import numpy as np

from cfdmod.adapters.memory import MemoryFieldStore
from cfdmod.core import ElementMeta, PointsDataSource, TimeAxis, Topology
from cfdmod.core.recipes import build_building_dynamic_response
from cfdmod.dynamics import read_eberick, read_tqs_portico

FIX = pathlib.Path(__file__).resolve().parents[2] / "fixtures" / "tests" / "dynamics" / "imports"
PORTICO = FIX / "portico"


def test_reads_floors_modes_frequencies_and_labels():
    sd = read_tqs_portico(PORTICO)
    assert sd.n_floors == 3
    assert sd.n_modes == 2
    np.testing.assert_allclose(np.asarray(sd.floor_points)[:, 2], [3.0, 6.0, 9.0])
    np.testing.assert_allclose(np.asarray(sd.natural_frequencies) / (2 * np.pi), [0.25, 0.60])
    # Floor names (tab-separated, may contain spaces) survive as metadata labels.
    assert sd.floor_labels == ["PAV 1", "PAV 2", "PAV 3"]


def test_unit_conversions_and_mass_normalization():
    sd = read_tqs_portico(PORTICO)
    np.testing.assert_allclose(sd.floors_radius, [5.0, 5.0, 5.0])
    np.testing.assert_allclose(sd.floors_mass[0], 0.30 * (9806.65 / 0.01))
    phi = np.asarray(sd.mode_shapes)
    m = np.asarray(sd.floors_mass)[:, None]
    r = np.asarray(sd.floors_radius)[:, None]
    m_gen = (m * (phi[:, :, 0] ** 2 + phi[:, :, 1] ** 2 + (r * phi[:, :, 2]) ** 2)).sum(axis=0)
    np.testing.assert_allclose(m_gen, np.ones(2), rtol=1e-9)


def test_matches_eberick_on_the_same_building():
    # The portico and eberick fixtures describe the same synthetic building.
    p = read_tqs_portico(PORTICO)
    e = read_eberick(FIX / "eberick")
    np.testing.assert_allclose(p.floors_mass, e.floors_mass, rtol=1e-9)
    np.testing.assert_allclose(p.floors_radius, e.floors_radius, rtol=1e-9)
    np.testing.assert_allclose(p.natural_frequencies, e.natural_frequencies, rtol=1e-9)
    np.testing.assert_allclose(p.mode_shapes, e.mode_shapes, rtol=1e-9, atol=1e-12)
    assert p.floor_labels == e.floor_labels


def test_explicit_file_paths():
    sd = read_tqs_portico(
        PORTICO,
        masses_file=PORTICO / "PORTICO_MASSAS_PAVIMENTO.TXT",
        modos_file=PORTICO / "PORTICO_MODOS_PAVIMENTO.TXT",
        modes_file=PORTICO / "modes.csv",
    )
    assert sd.n_floors == 3 and sd.n_modes == 2


def test_active_modes_selection():
    sd = read_tqs_portico(PORTICO, active_modes=[2])
    assert sd.n_modes == 1
    np.testing.assert_allclose(np.asarray(sd.natural_frequencies) / (2 * np.pi), [0.60])


def test_feeds_building_dynamic_recipe():
    sd = read_tqs_portico(PORTICO)
    cfg = sd.to_config(damping_ratio=0.02)
    n_floors, n_t, dt = sd.n_floors, 128, 0.05
    t = np.arange(n_t) * dt
    load = PointsDataSource(
        time=TimeAxis(initial_time=0.0, timestep_size=dt, n_timesteps=n_t),
        topology=Topology.points(np.asarray(sd.floor_points)),
        elements=ElementMeta(position=np.asarray(sd.floor_points)),
        fields=MemoryFieldStore(
            {
                "cf_x": 1e3 * np.sin(2 * np.pi * 0.2 * t) * np.ones((n_floors, n_t)),
                "cf_y": 5e2 * np.cos(2 * np.pi * 0.3 * t) * np.ones((n_floors, n_t)),
                "cm_z": 1e2 * np.sin(2 * np.pi * 0.25 * t) * np.ones((n_floors, n_t)),
            }
        ),
    )
    resp = build_building_dynamic_response(load, cfg)
    for name in ("disp_x", "disp_y", "rot_z", "feq_x", "feq_y", "meq_z"):
        arr = np.asarray(resp.fields.read(name))
        assert arr.shape == (n_floors, n_t)
        assert np.all(np.isfinite(arr))
