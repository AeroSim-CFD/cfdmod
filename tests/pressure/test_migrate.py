"""Smoke tests for cfdmod.pressure.migrate.

The migration helpers convert legacy pandas-HDFStore body/probe files into
the new XDMF+H5 layout. They aren't on the hot path of the v2 pipeline but
are still public, so coverage avoids silent regressions.
"""

from __future__ import annotations

import pathlib

import h5py
import numpy as np
import pandas as pd
import pytest
from lnas import LnasFormat, LnasGeometry

from cfdmod.io.xdmf import get_pressure_keys
from cfdmod.pressure.migrate import migrate_body_h5, migrate_probe_h5

pytestmark = pytest.mark.integration


def _write_legacy_body(path: pathlib.Path, n_tri: int = 4) -> tuple[np.ndarray, np.ndarray]:
    """Write a minimal legacy-format body H5 (pandas HDFStore with /step* keys)
    and return (time_steps, pressure_matrix) arrays for assertions."""
    times = np.array([0.0, 1.5], dtype=np.float64)
    pressures = np.array([[1.0, 2.0, 3.0, 4.0], [5.0, 6.0, 7.0, 8.0]], dtype=np.float32)

    with pd.HDFStore(path, mode="w") as store:
        for t, row in zip(times, pressures):
            df = pd.DataFrame(
                [
                    {
                        "time_step": t,
                        **{str(i): row[i] for i in range(n_tri)},
                    }
                ]
            )
            store.put(f"step{int(t):07d}", df, format="table")
    return times, pressures


def _write_lnas(path: pathlib.Path, n_tri: int = 4) -> None:
    triangles = np.arange(3 * n_tri, dtype=np.int32).reshape(n_tri, 3)
    vertices = np.random.default_rng(0).random((3 * n_tri, 3))
    fmt = LnasFormat(
        version="v0.5.2",
        geometry=LnasGeometry(vertices=vertices, triangles=triangles),
        surfaces={"all": np.arange(n_tri, dtype=np.int32)},
    )
    fmt.to_file(path)


def test_migrate_body_h5_produces_new_layout(tmp_path):
    legacy = tmp_path / "legacy_body.h5"
    new = tmp_path / "body.new.h5"
    mesh_path = tmp_path / "mesh.lnas"

    times, pressures = _write_legacy_body(legacy, n_tri=4)
    _write_lnas(mesh_path, n_tri=4)

    migrate_body_h5(legacy, mesh_path, new, macroscopic_type="pressure")

    keys = get_pressure_keys(new, "pressure")
    assert len(keys) == len(times)
    with h5py.File(new, "r") as f:
        assert f["Triangles"].shape == (4, 3)
        assert f["Geometry"].shape == (12, 3)
        for (_, key), expected in zip(keys, pressures):
            np.testing.assert_array_almost_equal(f["pressure"][key][:], expected)
        np.testing.assert_array_equal(f["meta"]["time_steps"][:], times)

    # Sibling .xdmf was written too
    assert new.with_suffix(".xdmf").exists()


def test_migrate_body_h5_rho_applies_cs2_scaling(tmp_path):
    legacy = tmp_path / "legacy.h5"
    new = tmp_path / "out.h5"
    mesh_path = tmp_path / "mesh.lnas"
    _, pressures = _write_legacy_body(legacy, n_tri=4)
    _write_lnas(mesh_path, n_tri=4)

    migrate_body_h5(legacy, mesh_path, new, macroscopic_type="rho")

    keys = get_pressure_keys(new, "pressure")
    with h5py.File(new, "r") as f:
        for (_, key), expected in zip(keys, pressures):
            np.testing.assert_array_almost_equal(f["pressure"][key][:], expected * (1.0 / 3.0))


def test_migrate_probe_h5(tmp_path):
    legacy = tmp_path / "legacy_probe.h5"
    new = tmp_path / "probe.new.h5"

    times = np.array([0.0, 0.5, 1.0])
    pressures = np.array([10.0, 20.0, 30.0])
    with pd.HDFStore(legacy, mode="w") as store:
        for t, p in zip(times, pressures):
            df = pd.DataFrame([{"time_step": t, "0": p}])
            store.put(f"step{int(t * 10):07d}", df, format="table")

    migrate_probe_h5(legacy, new, macroscopic_type="pressure")

    with h5py.File(new, "r") as f:
        keys = sorted(f["pressure"].keys(), key=lambda k: float(k[1:]))
        assert len(keys) == len(times)
        np.testing.assert_array_almost_equal(
            np.array([f["pressure"][k][0] for k in keys]), pressures
        )
        # Probe writes a trivial single-triangle, single-vertex placeholder.
        assert f["Triangles"].shape == (1, 3)
        assert f["Geometry"].shape == (1, 3)
    assert new.with_suffix(".xdmf").exists()
