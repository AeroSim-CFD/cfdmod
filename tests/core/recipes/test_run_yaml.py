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
def cp_template(tmp_path):
    """Copy the cp.yaml template + its data into tmp_path so the
    template's relative paths resolve, and the run writes into tmp_path."""
    template_dir = tmp_path / "templates"
    data_dir = tmp_path / "data"
    template_dir.mkdir()
    shutil.copytree(FIXTURES / "data", data_dir)
    shutil.copy(FIXTURES / "templates" / "cp.yaml", template_dir / "cp.yaml")
    return template_dir / "cp.yaml"


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
    assert np.all(np.abs(mean) < 5.0), f"Cp mean out of plausible range: {mean.min()}..{mean.max()}"
    assert np.all(rms >= 0)
    assert np.all(rms < 2.0)
