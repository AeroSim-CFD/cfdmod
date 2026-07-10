"""Smoke tests for the building dynamic-response plotting / export helpers.

Rendering runs on the headless Agg backend; the tests assert the helpers
build a figure / write a file against the v3 data sources without error.
"""

from __future__ import annotations

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from cfdmod.adapters.memory import MemoryFieldStore  # noqa: E402
from cfdmod.core import ElementMeta, PointsDataSource, TimeAxis, Topology  # noqa: E402
from cfdmod.core.container import Container  # noqa: E402
from cfdmod.core.recipes import (  # noqa: E402
    BuildingDynamicConfig,
    ComfortConfig,
    build_building_dynamic_response,
    build_point_accelerations,
)
from cfdmod.dynamics import BuildingCaseParameters, plotting  # noqa: E402

N_FLOORS = 3
N_MODES = 2
N_T = 256
DT = 0.05
FREQS_HZ = np.array([0.2, 0.5])


def _load_source():
    t = np.arange(N_T) * DT
    rows = [(1.0 + 0.2 * f) * np.sin(2 * np.pi * 0.3 * t + 0.1 * f) for f in range(N_FLOORS)]
    cf_x = np.vstack(rows)
    cf_y = 0.7 * cf_x
    cm_z = 0.2 * cf_x
    pts = np.zeros((N_FLOORS, 3))
    return PointsDataSource(
        time=TimeAxis(initial_time=0.0, timestep_size=DT, n_timesteps=N_T),
        topology=Topology.points(pts),
        elements=ElementMeta(position=pts),
        fields=MemoryFieldStore({"cf_x": cf_x, "cf_y": cf_y, "cm_z": cm_z}),
    )


def _response(load_source):
    phi = np.zeros((N_FLOORS, N_MODES, 3))
    phi[:, 0, 0] = np.linspace(0.1, 1.0, N_FLOORS)
    phi[:, 1, 1] = np.linspace(0.1, 1.0, N_FLOORS)
    phi[:, :, 2] = 0.01
    cfg = BuildingDynamicConfig(
        mode_shapes=phi,
        floor_points=np.zeros((N_FLOORS, 3)),
        cm_positions=np.tile([0.4, 0.2], (N_FLOORS, 1)),
        floors_mass=np.full(N_FLOORS, 120.0),
        floors_radius=np.full(N_FLOORS, 2.7),
        natural_frequencies=2 * np.pi * FREQS_HZ,
        damping_ratio=0.02,
    )
    return build_building_dynamic_response(load_source, cfg)


def test_plot_force_spectrum():
    fig, ax = plotting.plot_force_spectrum(_load_source(), FREQS_HZ)
    assert len(ax.get_lines()) > 0
    plt.close(fig)


def test_plot_displacement():
    resp = _response(_load_source())
    fig, ax = plotting.plot_displacement(resp, floor=N_FLOORS - 1, plot_limit=1.0)
    assert ax.get_xlabel() == "x [m]"
    plt.close(fig)


def test_plot_acceleration_floor_by_floor():
    resp = _response(_load_source())
    acc = build_point_accelerations(
        resp, ComfortConfig(cm_positions=np.tile([0.4, 0.2], (N_FLOORS, 1)), point=(2.0, 1.0))
    )
    peak = np.abs(np.asarray(acc.fields.read("acc_mag"))).max(axis=1)
    fig, ax = plotting.plot_acceleration_floor_by_floor(
        peak, float(FREQS_HZ.min()), rec_period=10, standards_to_use=["melbourne"]
    )
    assert len(ax.get_lines()) > 0
    plt.close(fig)


def test_plot_max_acceleration():
    fig, ax = plotting.plot_max_acceleration({1.0: 0.02, 10.0: 0.05}, FREQS_HZ)
    assert ax.get_title() == "Maximum acceleration"
    plt.close(fig)


def test_get_xlims_symmetric_and_rounded():
    lo, hi = plotting.get_xlims([-3.0, -1.0], [2.0, 4.0])
    assert lo == -hi
    assert hi >= 4.0


def test_plot_floor_by_floor_mean_peaks():
    vals = {
        stat: {ax: np.linspace(-1, 1, N_FLOORS) * (1 + i) for ax in ("x", "y", "z")}
        for i, stat in enumerate(("min", "max", "mean"))
    }
    fig, axs = plotting.plot_floor_by_floor_mean_peaks(
        vals_plot=vals, vals_labels=("Fx", "Fy", "Mz"), wind_dir=90.0, y_abs=None
    )
    assert axs.shape == (3,)
    plt.close(fig)


def _stats_df():
    directions = np.array([0.0, 30.0, 60.0])
    cols = {"direction": directions}
    for d in ("x", "y", "z"):
        cols[f"min_{d}"] = -np.ones(3)
        cols[f"max_{d}"] = np.ones(3)
        cols[f"mean_{d}"] = np.zeros(3)
    return pd.DataFrame(cols)


def test_plot_global_stats_per_direction_and_csv(tmp_path):
    stats = {
        "forces_static": _stats_df(),
        "forces_static_eq": _stats_df(),
        "moments_static": _stats_df(),
        "moments_static_eq": _stats_df(),
    }
    fig, axs = plotting.plot_global_stats_per_direction({0.01: stats})
    assert axs.shape == (3, 2)
    plt.close(fig)

    csv_path = tmp_path / "stats.csv"
    plotting.export_global_stats_per_direction_csv(csv_path, stats)
    back = pd.read_csv(csv_path)
    assert "forces_static_max_x" in back.columns


def test_effective_loads_and_eberick_export(tmp_path):
    load = _load_source()
    resp = _response(load)
    cases = [
        BuildingCaseParameters(direction=0.0, xi=0.02, recurrence_period=50.0),
        BuildingCaseParameters(direction=90.0, xi=0.02, recurrence_period=50.0),
    ]
    container = Container(items={c: resp for c in cases})

    tables = plotting.effective_peak_loads_per_direction(container)
    assert set(tables) == {"Fx", "Fy", "Mz"}
    assert list(tables["Fx"].columns) == ["0.0", "90.0"]

    heights = np.arange(1, N_FLOORS + 1) * 3.0
    names = [f"P{i}" for i in range(N_FLOORS)]
    for df in tables.values():
        plotting.add_eberick_floor_height_and_pavements(df, heights, names)

    out = tmp_path / "eberick.xlsx"
    plotting.export_eberick_tables_to_xlsx(tables, out)
    assert out.exists()
    back = pd.read_excel(out, sheet_name="Fx")
    assert "Pavimento" in back.columns
