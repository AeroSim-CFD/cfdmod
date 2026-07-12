"""Tests for cfdmod.report.DebugWriter and the plot_config tech style."""

from __future__ import annotations

import matplotlib

matplotlib.use("Agg")  # headless

import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402
import pytest  # noqa: E402

from cfdmod import plot_config  # noqa: E402
from cfdmod.report import DebugWriter  # noqa: E402

pytestmark = pytest.mark.unit


def test_save_csv_writes_and_roundtrips(tmp_path):
    dbg = DebugWriter(tmp_path, stage="cf", version="v1")
    df = pd.DataFrame({"floor": [1, 2], "cf_x": [0.5, 0.7]})
    path = dbg.save_csv(df, "tables/loads.csv")
    assert path.exists()
    assert path == tmp_path / "debug" / "v1" / "cf" / "tables" / "loads.csv"
    back = pd.read_csv(path)
    assert list(back.columns) == ["floor", "cf_x"]
    assert back.shape == (2, 2)


def test_save_csv_deliverable_root(tmp_path):
    dbg = DebugWriter(tmp_path, stage="cf", version="v1")
    path = dbg.save_csv(pd.DataFrame({"a": [1]}), "x.csv", deliverable=True)
    assert path == tmp_path / "deliverables" / "v1" / "cf" / "x.csv"


def test_skip_if_exists_does_not_overwrite(tmp_path):
    dbg = DebugWriter(tmp_path, stage="cf", version="v1")
    path = dbg.save_csv(pd.DataFrame({"a": [1]}), "x.csv")
    original = path.read_text()
    # Second call with different content but skip_if_exists must NOT overwrite.
    dbg.save_csv(pd.DataFrame({"a": [999]}), "x.csv", skip_if_exists=True)
    assert path.read_text() == original


def test_savefig_skip_if_exists(tmp_path):
    dbg = DebugWriter(tmp_path, stage="inflow", version="v1")
    fig = plt.figure()
    p1 = dbg.savefig(fig, "f.png")
    mtime = p1.stat().st_mtime_ns
    dbg.savefig(fig, "f.png", skip_if_exists=True)
    assert p1.stat().st_mtime_ns == mtime  # unchanged
    plt.close(fig)


def test_set_style_tech_applies_marker_style():
    plot_config.set_style_tech()
    assert plt.rcParams["lines.linestyle"] == ""
    assert plt.rcParams["lines.markersize"] == 6
