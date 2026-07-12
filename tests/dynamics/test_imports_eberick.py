"""Tests for the Eberick per-floor reader, cross-checked against TQS."""

from __future__ import annotations

import pathlib

import numpy as np

from cfdmod.dynamics import read_eberick, read_tqs_portels

FIX = pathlib.Path(__file__).resolve().parents[2] / "fixtures" / "tests" / "dynamics" / "imports"


def test_reads_periods_geometry_and_shapes():
    sd = read_eberick(FIX / "eberick" / "modal.xlsx")
    assert sd.n_floors == 3
    assert sd.n_modes == 2
    np.testing.assert_allclose(np.asarray(sd.floor_points)[:, 2], [3.0, 6.0, 9.0])
    np.testing.assert_allclose(np.asarray(sd.natural_frequencies) / (2 * np.pi), [1.0, 2.5])


def test_mode_shapes_are_mass_normalized():
    sd = read_eberick(FIX / "eberick" / "modal.xlsx")
    phi = np.asarray(sd.mode_shapes)
    m = np.asarray(sd.floors_mass)[:, None]
    r = np.asarray(sd.floors_radius)[:, None]
    m_gen = (m * (phi[:, :, 0] ** 2 + phi[:, :, 1] ** 2 + (r * phi[:, :, 2]) ** 2)).sum(axis=0)
    np.testing.assert_allclose(m_gen, np.ones(2), rtol=1e-9)


def test_eberick_matches_tqs_on_the_same_building():
    # The two fixtures describe the same synthetic 3-floor building, so the
    # readers must agree on geometry, frequencies and (up to sign) shapes.
    eb = read_eberick(FIX / "eberick" / "modal.xlsx")
    tqs = read_tqs_portels(FIX / "tqs")
    np.testing.assert_allclose(eb.floors_mass, tqs.floors_mass, rtol=1e-9)
    np.testing.assert_allclose(eb.floors_radius, tqs.floors_radius, rtol=1e-9)
    np.testing.assert_allclose(eb.cm_positions, tqs.cm_positions, atol=1e-9)
    np.testing.assert_allclose(eb.natural_frequencies, tqs.natural_frequencies, rtol=1e-9)
    np.testing.assert_allclose(eb.mode_shapes, tqs.mode_shapes, rtol=1e-9, atol=1e-12)


def test_active_modes_selection():
    sd = read_eberick(FIX / "eberick" / "modal.xlsx", active_modes=[2])
    assert sd.n_modes == 1
    np.testing.assert_allclose(np.asarray(sd.natural_frequencies) / (2 * np.pi), [2.5])
