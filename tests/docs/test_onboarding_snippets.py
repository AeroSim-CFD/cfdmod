"""Execute the code snippets from the Getting Started docs end to end.

These tests pin the copy-pasteable examples in
``docs/source/getting_started/index.rst`` and ``reading_outputs.rst`` so
they cannot drift from the API silently. Each test mirrors a snippet as
closely as the fixture allows.
"""

from __future__ import annotations

import pathlib
import shutil

import matplotlib
import pytest

matplotlib.use("Agg")  # headless: no display needed to exercise plot_timeseries

pytestmark = pytest.mark.integration

FIXTURES = pathlib.Path(__file__).parents[2] / "fixtures" / "tests" / "pressure"


@pytest.fixture
def pressure_workdir(tmp_path):
    """Copy templates + data into a writable scratch dir (as the docs
    instruct repository-checkout users to do)."""
    template_dir = tmp_path / "templates"
    template_dir.mkdir()
    shutil.copytree(FIXTURES / "data", tmp_path / "data")
    shutil.copytree(FIXTURES / "galpao", tmp_path / "galpao")
    for tpl in (FIXTURES / "templates").iterdir():
        shutil.copy(tpl, template_dir / tpl.name)
    return template_dir


def test_getting_started_first_cp(pressure_workdir):
    """Getting Started -> "Run your first Cp" (Python snippet)."""
    from cfdmod import XdmfH5Storage, load_template, run_template

    template = load_template(pressure_workdir / "cp.yaml")
    bindings = run_template(template, storage=XdmfH5Storage(root="."))
    cp_t = bindings["cp_t"]
    cp_series = cp_t.fields.read("cp")

    assert cp_t.kind == "surface"
    assert cp_series.shape[1] == 101  # 101 timesteps in the fixture

    out_dir = pressure_workdir / "out"
    assert (out_dir / "cp.time_series.h5").exists()
    assert (out_dir / "cp.time_series.xdmf").exists()
    assert (out_dir / "cp.stats.h5").exists()
    assert (out_dir / "cp.stats.xdmf").exists()


def test_reading_outputs_pandas_csv_plot(pressure_workdir):
    """Reading Outputs -> pandas / CSV / quick-plot snippets."""
    from cfdmod import (
        XdmfH5Storage,
        load_template,
        plot_timeseries,
        read_timeseries_df,
        run_template,
        to_csv,
    )

    run_template(load_template(pressure_workdir / "cp.yaml"), storage=XdmfH5Storage(root="."))
    ts_h5 = pressure_workdir / "out" / "cp.time_series.h5"

    df = read_timeseries_df(ts_h5, "cp", triangles=[0, 1, 2])
    assert df.index.name == "time_normalized"
    assert list(df.columns) == [0, 1, 2]
    assert df.shape == (101, 3)

    csv_path = pressure_workdir / "out" / "cp_selected.csv"
    to_csv(df, csv_path)
    assert csv_path.exists()

    ax = plot_timeseries(df, title="Cp on selected triangles", ylabel="Cp")
    assert ax.get_ylabel() == "Cp"


def test_reading_outputs_metadata_roundtrip(pressure_workdir):
    """Reading Outputs -> reproducibility-metadata round-trip snippet."""
    from cfdmod import (
        XdmfH5Storage,
        load_template,
        read_processing_metadata,
        run_template,
        write_processing_metadata,
    )

    run_template(load_template(pressure_workdir / "cp.yaml"), storage=XdmfH5Storage(root="."))
    stats_h5 = pressure_workdir / "out" / "cp.stats.h5"

    write_processing_metadata(
        stats_h5,
        "stats",
        {"note": "galpao Cp demo", "dynamic_pressure_factor": 800.0},
    )
    meta = read_processing_metadata(stats_h5, "stats")

    assert meta["config"] == {"note": "galpao Cp demo", "dynamic_pressure_factor": 800.0}
    assert "cfdmod_version" in meta
    assert "produced_at" in meta
