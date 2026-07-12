"""Tests for the Eberick per-floor reader (real 2-file layout)."""

from __future__ import annotations

import pathlib

import numpy as np

from cfdmod.adapters.memory import MemoryFieldStore
from cfdmod.core import ElementMeta, PointsDataSource, TimeAxis, Topology
from cfdmod.core.recipes import build_building_dynamic_response
from cfdmod.dynamics import EberickUnits, read_eberick

FIX = pathlib.Path(__file__).resolve().parents[2] / "fixtures" / "tests" / "dynamics" / "imports"
EB = FIX / "eberick"


def test_reads_floors_modes_and_frequencies():
    sd = read_eberick(EB)
    assert sd.n_floors == 3
    assert sd.n_modes == 2
    # Elevations converted cm -> m; frequencies taken straight from the FORMAS blocks.
    np.testing.assert_allclose(np.asarray(sd.floor_points)[:, 2], [3.0, 6.0, 9.0])
    np.testing.assert_allclose(np.asarray(sd.natural_frequencies) / (2 * np.pi), [0.25, 0.60])


def test_unit_conversions():
    sd = read_eberick(EB)
    # radius = sqrt(I/M) = sqrt(75000/0.30) = 500 cm -> 5 m; CoM (50, 30) cm -> (0.5, 0.3) m.
    np.testing.assert_allclose(sd.floors_radius, [5.0, 5.0, 5.0])
    np.testing.assert_allclose(np.asarray(sd.cm_positions)[0], [0.5, 0.3])
    # mass 0.30 tf.s^2/cm -> 0.30 * 980665 kg.
    np.testing.assert_allclose(sd.floors_mass[0], 0.30 * (9806.65 / 0.01))


def test_mode_shapes_are_mass_normalized():
    sd = read_eberick(EB)
    phi = np.asarray(sd.mode_shapes)
    m = np.asarray(sd.floors_mass)[:, None]
    r = np.asarray(sd.floors_radius)[:, None]
    m_gen = (m * (phi[:, :, 0] ** 2 + phi[:, :, 1] ** 2 + (r * phi[:, :, 2]) ** 2)).sum(axis=0)
    np.testing.assert_allclose(m_gen, np.ones(2), rtol=1e-9)


def test_active_modes_selection():
    sd = read_eberick(EB, active_modes=[2])
    assert sd.n_modes == 1
    np.testing.assert_allclose(np.asarray(sd.natural_frequencies) / (2 * np.pi), [0.60])


def test_custom_units_are_honoured():
    # With identity units the elevations stay in the raw (cm) magnitude.
    sd = read_eberick(EB, units=EberickUnits(length_to_m=1.0, mass_to_kg=1.0))
    np.testing.assert_allclose(np.asarray(sd.floor_points)[:, 2], [300.0, 600.0, 900.0])
    np.testing.assert_allclose(sd.floors_mass[0], 0.30)


def test_feeds_building_dynamic_recipe():
    sd = read_eberick(EB)
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
