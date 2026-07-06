"""Integration tests for the YAML-driven runner against real fixtures."""

from __future__ import annotations

import pathlib
import shutil

import h5py
import numpy as np
import pytest

from cfdmod import load_template, run_template
from cfdmod.adapters.xdmf_h5 import XdmfH5Storage

FIXTURES = pathlib.Path(__file__).parents[3] / "fixtures" / "tests" / "pressure"


@pytest.fixture
def pressure_workdir(tmp_path):
    """Copy the templates + all fixture data into tmp_path so each
    test has its own writable scratch space without polluting the
    repo's fixtures directory."""
    template_dir = tmp_path / "templates"
    data_dir = tmp_path / "data"
    galpao_dir = tmp_path / "galpao"
    template_dir.mkdir()
    shutil.copytree(FIXTURES / "data", data_dir)
    shutil.copytree(FIXTURES / "galpao", galpao_dir)
    for tpl in (FIXTURES / "templates").iterdir():
        shutil.copy(tpl, template_dir / tpl.name)
    return template_dir


@pytest.fixture
def cp_template(pressure_workdir):
    return pressure_workdir / "cp.yaml"


@pytest.fixture
def cf_template(pressure_workdir):
    return pressure_workdir / "cf.yaml"


@pytest.fixture
def cm_template(pressure_workdir):
    return pressure_workdir / "cm.yaml"


@pytest.fixture
def ce_template(pressure_workdir):
    return pressure_workdir / "ce.yaml"


def test_cp_template_runs_against_real_fixtures(cp_template):
    template = load_template(cp_template)
    storage = XdmfH5Storage(pathlib.Path("/"))
    bindings = run_template(template, storage=storage)

    assert "cp_t" in bindings
    cp_t = bindings["cp_t"]
    assert cp_t.kind == "surface"
    assert cp_t.n_elements == 2915

    stats = bindings["cp_stats"]
    assert stats.time.is_time_aggregated
    assert sorted(stats.field_names) == ["max", "mean", "min", "rms"]


def test_cp_template_writes_xdmf_and_h5(cp_template):
    template = load_template(cp_template)
    run_template(template, storage=XdmfH5Storage(pathlib.Path("/")))

    out_dir = cp_template.parent / "out"
    assert (out_dir / "cp.time_series.h5").exists()
    assert (out_dir / "cp.time_series.xdmf").exists()
    assert (out_dir / "cp.stats.h5").exists()
    assert (out_dir / "cp.stats.xdmf").exists()


def test_cp_template_output_matches_expected_shape(cp_template):
    template = load_template(cp_template)
    run_template(template, storage=XdmfH5Storage(pathlib.Path("/")))

    out_dir = cp_template.parent / "out"
    with h5py.File(out_dir / "cp.time_series.h5", "r") as f:
        assert "Triangles" in f
        assert "Geometry" in f
        assert "cp" in f
        # 101 timesteps in the fixture.
        time_keys = list(f["cp"].keys())
        assert len(time_keys) == 101
    with h5py.File(out_dir / "cp.stats.h5", "r") as f:
        # stats output stored under /stats/ when the field name has no slash.
        assert "stats" in f
        keys = set(f["stats"].keys()) - {"Geometry", "Triangles"}
        assert keys >= {"max", "mean", "min", "rms"}


def test_cp_template_results_are_in_plausible_range(cp_template):
    """Cp typical range for the galpao fixture is roughly [-2, 1]."""
    template = load_template(cp_template)
    bindings = run_template(template, storage=XdmfH5Storage(pathlib.Path("/")))

    stats = bindings["cp_stats"]
    mean = stats.fields.read("mean")
    rms = stats.fields.read("rms")
    assert np.all(
        np.abs(mean) < 5.0
    ), f"Cp mean out of plausible range: {mean.min()}..{mean.max()}"
    assert np.all(rms >= 0)
    assert np.all(rms < 2.0)


def test_cf_template_runs_after_cp(cp_template, cf_template):
    """End-to-end Cp -> Cf chain: Cp template writes cp.time_series, Cf
    template reads it back, attaches the mesh, and produces per-body
    cf_x/cf_y/cf_z fields."""
    storage = XdmfH5Storage(pathlib.Path("/"))

    cp_tpl = load_template(cp_template)
    run_template(cp_tpl, storage=storage)

    cf_tpl = load_template(cf_template)
    bindings = run_template(cf_tpl, storage=storage)

    # The Cf macro produces a GroupsDataSource (one row per body) for
    # each direction; the storage broadcasts back to the parent surface
    # on write.
    cf_x = bindings["cf_x"]
    assert cf_x.kind == "groups"
    # Only one body declared in the template -> n_elements == 1.
    assert cf_x.n_elements == 1
    cf_x_series = cf_x.fields.read("cf_x")
    assert cf_x_series.shape[0] == 1  # one body
    assert cf_x_series.shape[1] == 101  # 101 timesteps in the fixture


def test_cf_template_writes_broadcast_surface_h5(cp_template, cf_template):
    storage = XdmfH5Storage(pathlib.Path("/"))
    run_template(load_template(cp_template), storage=storage)
    run_template(load_template(cf_template), storage=storage)

    out_dir = cp_template.parent / "out"
    for direction in ("x", "y", "z"):
        h5 = out_dir / f"cf_{direction}.time_series.h5"
        assert h5.exists(), f"missing {h5}"
        with h5py.File(h5, "r") as f:
            assert "Triangles" in f
            assert "Geometry" in f
            # The broadcast surface has 2915 triangles (same as parent mesh).
            assert f["Triangles"].shape[0] == 2915
            group = f[f"cf_{direction}"]
            time_keys = list(group.keys())
            assert len(time_keys) == 101
            sample = group[time_keys[0]][:]
            # Every triangle in the lone body gets the same value (broadcast).
            unique_vals = np.unique(sample[~np.isnan(sample)])
            assert unique_vals.size == 1


def test_cm_template_produces_per_body_moment(cp_template, cm_template):
    storage = XdmfH5Storage(pathlib.Path("/"))
    run_template(load_template(cp_template), storage=storage)
    bindings = run_template(load_template(cm_template), storage=storage)

    cm_x = bindings["cm_x"]
    assert cm_x.kind == "groups"
    assert cm_x.n_elements == 1  # one body in the template
    cm_x_series = cm_x.fields.read("cm_x")
    assert cm_x_series.shape == (1, 101)


def test_ce_template_produces_per_zone_values(cp_template, ce_template):
    storage = XdmfH5Storage(pathlib.Path("/"))
    run_template(load_template(cp_template), storage=storage)
    bindings = run_template(load_template(ce_template), storage=storage)

    ce = bindings["ce"]
    assert ce.kind == "groups"
    # 8 zones in a 2x2x2 grid; some may have no triangles -> ungrouped.
    assert 1 <= ce.n_elements <= 8
    ce_series = ce.fields.read("ce")
    assert ce_series.shape[1] == 101
    # area-weighted means of Cp should land in a plausible range.
    assert np.all(np.abs(ce_series) < 5.0)
