"""Generate the high-rise stage notebooks (clean, no stored outputs).

Run: uv run python examples/high_rise/_build_notebooks.py
Writes 01_inflow, 02_cp, 03_cf, 04_dynamic, 05_facade, 06_structure next to
this script.

The notebooks are thin drivers: config is read from environment variables with
in-repo fixture defaults, so they run headless (nbconvert / _validate_notebooks)
without any external data, and point at a real case by setting the CFDMOD_HR_*
variables. All reusable logic lives in the pp/ helper package.
"""

from __future__ import annotations

import pathlib

import nbformat
from nbformat.v4 import new_code_cell, new_markdown_cell, new_notebook

HERE = pathlib.Path(__file__).resolve().parent

SETUP = """\
import os
import pathlib
import sys

import numpy as np  # noqa: F401  (used across later cells)


def _find_repo(start: pathlib.Path) -> pathlib.Path:
    p = start.resolve()
    while p != p.parent:
        if (p / "pyproject.toml").exists():
            return p
        p = p.parent
    return start.resolve()


# pp is a notebook-side package, imported after inserting its dir on sys.path.
REPO = _find_repo(pathlib.Path.cwd())
sys.path.insert(0, str(REPO / "examples" / "high_rise"))

import pp  # noqa: E402
from pp import plotting  # noqa: E402

plotting.apply_style()

FIX = REPO / "fixtures" / "tests"
OUTPUT_BASE = pathlib.Path(
    os.environ.get("CFDMOD_HR_OUTPUT_BASE", REPO / "examples" / "high_rise" / "_run")
)
VERSION = os.environ.get("CFDMOD_HR_VERSION", "example")
print("REPO:", REPO)
print("OUTPUT_BASE:", OUTPUT_BASE, "| VERSION:", VERSION)"""


# --------------------------------------------------------------------------
# 01 -- inflow validation
# --------------------------------------------------------------------------

INFLOW_CELLS = [
    new_markdown_cell(
        "# High-rise 01 - Inlet profile validation\n"
        "\n"
        "Validate the atmospheric boundary layer: detect the vertical profiles in the\n"
        "probe cloud, plot mean velocity / turbulence intensity / spectrum into `debug/`,\n"
        "and extract the simulation mean velocity at the reference height (`u_ref`). That\n"
        "`u_ref` feeds the Cp non-dimensionalisation in notebook 02.\n"
        "\n"
        "Defaults run on the `pitot_inlet` fixture. Point at a real case by setting the\n"
        "`CFDMOD_HR_INFLOW_HIST` / `CFDMOD_HR_INFLOW_POINTS` / `CFDMOD_HR_REF_HEIGHT`\n"
        "environment variables (or editing the config cell)."
    ),
    new_code_cell(SETUP),
    new_code_cell(
        "from cfdmod.inflow import InflowData, NormalizationParameters\n"
        "\n"
        "# --- config -------------------------------------------------------------\n"
        "HIST = pathlib.Path(\n"
        "    os.environ.get(\n"
        '        "CFDMOD_HR_INFLOW_HIST", FIX / "inflow" / "pitot_inlet" / "hist_series.csv"\n'
        "    )\n"
        ")\n"
        "POINTS = pathlib.Path(\n"
        '    os.environ.get("CFDMOD_HR_INFLOW_POINTS", FIX / "inflow" / "pitot_inlet" / "points.csv")\n'
        ")\n"
        'REFERENCE_HEIGHT = float(os.environ.get("CFDMOD_HR_REF_HEIGHT", "2.0"))\n'
        'COMPONENT = os.environ.get("CFDMOD_HR_COMPONENT", "ux")\n'
        "\n"
        'dbg = pp.DebugWriter(OUTPUT_BASE, stage="inflow", version=VERSION)\n'
        "inflow = InflowData.from_files(HIST, POINTS)\n"
        "profiles = pp.detect_profiles(inflow, min_points=3)\n"
        'print(f"detected {len(profiles)} vertical profile(s)")\n'
        "for p in profiles:\n"
        '    print(f"  {p.name}: {len(p.point_idx)} points, z in [{p.z.min():.2f}, {p.z.max():.2f}]")'
    ),
    new_code_cell(
        "# --- per-profile figures + reference velocity ---------------------------\n"
        "u_ref_by_profile = {}\n"
        "for prof in profiles:\n"
        "    u_ref = pp.reference_velocity(prof, inflow, REFERENCE_HEIGHT, component=COMPONENT)\n"
        "    u_ref_by_profile[prof.name] = u_ref\n"
        "    L = pp.inflow_report.integral_length_scale(\n"
        "        inflow, prof.nearest_index(REFERENCE_HEIGHT), u_ref, component=COMPONENT\n"
        "    )\n"
        '    print(f"{prof.name}: u_ref(z={REFERENCE_HEIGHT:g}) = {u_ref:.4f} m/s | L = {L:.4g} m")\n'
        "\n"
        "    norm = NormalizationParameters(\n"
        "        reference_velocity=max(u_ref, 1e-9), characteristic_length=1.0\n"
        "    )\n"
        "    for name, fig in {\n"
        '        "mean_velocity": pp.inflow_report.plot_mean_velocity(prof, inflow, component=COMPONENT),\n'
        '        "turbulence_intensity": pp.inflow_report.plot_turbulence_intensity(\n'
        "            prof, inflow, component=COMPONENT\n"
        "        ),\n"
        '        "spectrum": pp.inflow_report.plot_spectrum(\n'
        "            prof, inflow, REFERENCE_HEIGHT, norm, component=COMPONENT\n"
        "        ),\n"
        "    }.items():\n"
        '        dbg.savefig(fig, f"{prof.name}/{name}.png")\n'
        "        plotting.close(fig)\n"
        'print("figures written under", dbg.debug_dir)'
    ),
    new_code_cell(
        "import json\n"
        "\n"
        "# --- persist u_ref for the Cp step (the 'update config' step) -----------\n"
        "# The richest profile is the representative inlet column.\n"
        "primary = profiles[0].name if profiles else None\n"
        "u_ref = u_ref_by_profile.get(primary)\n"
        'out = dbg.deliverable_path("reference_velocity.json")\n'
        "out.write_text(\n"
        "    json.dumps(\n"
        '        {"profile": primary, "reference_height": REFERENCE_HEIGHT, "u_ref": u_ref},\n'
        "        indent=2,\n"
        "    )\n"
        ")\n"
        'print(f"u_ref = {u_ref} m/s -> {out}")\n'
        "# In notebook 02: case = case.with_reference_velocity(u_ref)"
    ),
]


# --------------------------------------------------------------------------
# 02 -- Cp
# --------------------------------------------------------------------------

CP_CELLS = [
    new_markdown_cell(
        "# High-rise 02 - Pressure coefficient (Cp)\n"
        "\n"
        "Compute `Cp = (p - p_ref) / q`, with `q = 0.5 * rho * U_H^2` from the case (using\n"
        "the `u_ref` measured in notebook 01). Writes the Cp time series to storage for\n"
        "notebook 03 and a stats summary to `deliverables/`.\n"
        "\n"
        "Defaults run on the `galpao` fixture (`u_h=0.05`, `rho=1.0` -> `q=0.00125`, matching\n"
        "the `cp.yaml` template). For a real case, set `CFDMOD_HR_CASE_DATA` to a `case_data`\n"
        "dir and `CFDMOD_HR_DATA_DIR` / `CFDMOD_HR_BODY_KEY` / `CFDMOD_HR_REF_KEY` to the data."
    ),
    new_code_cell(SETUP),
    new_code_cell(
        "import json\n"
        "\n"
        "from cfdmod.adapters.xdmf_h5 import XdmfH5Storage\n"
        "\n"
        "# --- config -------------------------------------------------------------\n"
        'DATA_DIR = pathlib.Path(os.environ.get("CFDMOD_HR_DATA_DIR", FIX / "pressure" / "data"))\n'
        'BODY_KEY = os.environ.get("CFDMOD_HR_BODY_KEY", "bodies.galpao")\n'
        'REF_KEY = os.environ.get("CFDMOD_HR_REF_KEY", "points.static_pressure")\n'
        "MESH = pathlib.Path(\n"
        '    os.environ.get("CFDMOD_HR_MESH", FIX / "pressure" / "galpao" / "galpao.normalized.lnas")\n'
        ")\n"
        "\n"
        "# Build the case: from a real case_data dir, or an example tuned to the mesh.\n"
        'CASE_DATA = os.environ.get("CFDMOD_HR_CASE_DATA")\n'
        'PARAMS = os.environ.get("CFDMOD_HR_PARAMS", "params_cat3.yaml")\n'
        "if CASE_DATA:\n"
        "    case = pp.HighRiseCase.from_case_data(pathlib.Path(CASE_DATA), PARAMS)\n"
        "else:\n"
        "    case = pp.example_high_rise_case(MESH)\n"
        "\n"
        "# Apply the reference velocity measured in notebook 01, if available.\n"
        'ref_json = OUTPUT_BASE / "deliverables" / VERSION / "inflow" / "reference_velocity.json"\n'
        'u_env = os.environ.get("CFDMOD_HR_UH")\n'
        "if u_env:\n"
        "    case = case.with_reference_velocity(float(u_env))\n"
        "elif ref_json.exists():\n"
        '    u = json.loads(ref_json.read_text()).get("u_ref")\n'
        "    if u:\n"
        "        case = case.with_reference_velocity(float(u))\n"
        'print(f"U_H = {case.u_h:.5g} m/s -> dynamic pressure q = {case.dynamic_pressure:.5g} Pa")'
    ),
    new_code_cell(
        "# --- compute Cp ---------------------------------------------------------\n"
        "storage = XdmfH5Storage(DATA_DIR)\n"
        "body = storage.read_data_source(pathlib.Path(BODY_KEY))\n"
        "p_ref = storage.read_data_source(pathlib.Path(REF_KEY))\n"
        'print(f"body: {body.kind}, {body.n_elements} elements, {body.time.n_timesteps} steps")\n'
        "\n"
        "cp = pp.cp_from_pressure(body, p_ref, case)\n"
        'cp_stats = pp.cp_from_pressure(body, p_ref, case, statistics=["mean", "rms", "min", "max"])\n'
        'mean_cp = cp_stats.fields.read("mean")\n'
        'print(f"Cp mean range: [{np.nanmin(mean_cp):.3f}, {np.nanmax(mean_cp):.3f}]")'
    ),
    new_code_cell(
        "# --- write Cp time series to storage for notebook 03 --------------------\n"
        'ARTIFACTS = OUTPUT_BASE / "artifacts" / VERSION\n'
        "ARTIFACTS.mkdir(parents=True, exist_ok=True)\n"
        "art_storage = XdmfH5Storage(ARTIFACTS)\n"
        'art_storage.write_data_source(pathlib.Path("cp.time_series"), cp)\n'
        'print("wrote", ARTIFACTS / "cp.time_series.h5")'
    ),
    new_code_cell(
        "import pandas as pd\n"
        "\n"
        "# --- Cp stats summary -> deliverables -----------------------------------\n"
        'dbg = pp.DebugWriter(OUTPUT_BASE, stage="cp", version=VERSION)\n'
        'fig, ax = plotting.new_axes(xlabel="mean Cp [-]", ylabel="count", title="Cp mean distribution")\n'
        "ax.hist(mean_cp[np.isfinite(mean_cp)], bins=40)\n"
        'dbg.savefig(fig, "cp_mean_hist.png")\n'
        "plotting.close(fig)\n"
        "\n"
        'stat_names = ["mean", "rms", "min", "max"]\n'
        "summary = pd.DataFrame(\n"
        "    {\n"
        '        "stat": stat_names,\n'
        '        "field_mean": [float(np.nanmean(cp_stats.fields.read(s))) for s in stat_names],\n'
        "    }\n"
        ")\n"
        'summary.to_csv(dbg.deliverable_path("cp_summary.csv"), index=False)\n'
        "summary"
    ),
]


# --------------------------------------------------------------------------
# 03 -- per-floor Cf / Cm
# --------------------------------------------------------------------------

CF_CELLS = [
    new_markdown_cell(
        "# High-rise 03 - Per-floor force / moment coefficients (Cf, Cm)\n"
        "\n"
        "Read the Cp time series from notebook 02, slice the body by floor z-edges, and sum\n"
        "the per-triangle force / moment contributions per floor (explicit reference-area\n"
        "normalisation). Writes per-floor Cf/Cm profiles to `debug/` and a load table to\n"
        "`deliverables/`.\n"
        "\n"
        "Uses the same case + mesh as notebook 02. For the fixture the floor edges come from\n"
        "the mesh z-range; for a real case they come from the case_data `HEIGHTS`."
    ),
    new_code_cell(SETUP),
    new_code_cell(
        "from cfdmod.adapters.xdmf_h5 import XdmfH5Storage\n"
        "\n"
        "# --- config -------------------------------------------------------------\n"
        "MESH = pathlib.Path(\n"
        '    os.environ.get("CFDMOD_HR_MESH", FIX / "pressure" / "galpao" / "galpao.normalized.lnas")\n'
        ")\n"
        'CASE_DATA = os.environ.get("CFDMOD_HR_CASE_DATA")\n'
        'PARAMS = os.environ.get("CFDMOD_HR_PARAMS", "params_cat3.yaml")\n'
        "if CASE_DATA:\n"
        "    case = pp.HighRiseCase.from_case_data(pathlib.Path(CASE_DATA), PARAMS)\n"
        "else:\n"
        "    case = pp.example_high_rise_case(MESH)\n"
        "\n"
        'ARTIFACTS = OUTPUT_BASE / "artifacts" / VERSION\n'
        'cp = XdmfH5Storage(ARTIFACTS).read_data_source(pathlib.Path("cp.time_series"))\n'
        'print(f"cp: {cp.n_elements} elements, {cp.time.n_timesteps} steps | {case.n_floors} floors")'
    ),
    new_code_cell(
        "# --- per-floor Cf / Cm --------------------------------------------------\n"
        'cf = pp.cf_per_floor(cp, str(MESH), case, directions=("x", "y"))\n'
        'cm = pp.cm_per_floor(cp, str(MESH), case, directions=("z",))\n'
        "\n"
        "# Floor mid-heights for plotting.\n"
        "edges = np.asarray(case.floor_heights)\n"
        "z_mid = 0.5 * (edges[:-1] + edges[1:])\n"
        'cfx_mean = np.nanmean(cf.fields.read("cf_x"), axis=1)\n'
        'cfy_mean = np.nanmean(cf.fields.read("cf_y"), axis=1)\n'
        'cmz_mean = np.nanmean(cm.fields.read("cm_z"), axis=1)\n'
        'print("per-floor mean Cf_x:", np.round(cfx_mean, 4))'
    ),
    new_code_cell(
        "import pandas as pd\n"
        "\n"
        "# --- figures + load table ----------------------------------------------\n"
        'dbg = pp.DebugWriter(OUTPUT_BASE, stage="cf", version=VERSION)\n'
        'fig, ax = plotting.new_axes(xlabel="mean coefficient [-]", ylabel="z [m]", title="Per-floor Cf / Cm")\n'
        'ax.plot(cfx_mean, z_mid, "-o", ms=3, label="Cf_x")\n'
        'ax.plot(cfy_mean, z_mid, "-o", ms=3, label="Cf_y")\n'
        'ax.plot(cmz_mean, z_mid, "-s", ms=3, label="Cm_z")\n'
        "ax.legend()\n"
        'dbg.savefig(fig, "per_floor_coefficients.png")\n'
        "plotting.close(fig)\n"
        "\n"
        "table = pd.DataFrame(\n"
        "    {\n"
        '        "floor": np.arange(len(z_mid)),\n'
        '        "z_mid": z_mid,\n'
        '        "cf_x_mean": cfx_mean,\n'
        '        "cf_y_mean": cfy_mean,\n'
        '        "cm_z_mean": cmz_mean,\n'
        "    }\n"
        ")\n"
        'table.to_csv(dbg.deliverable_path("per_floor_loads.csv"), index=False)\n'
        "table"
    ),
]


# --------------------------------------------------------------------------
# 04 -- dynamic response (HFPI / SDOF)
# --------------------------------------------------------------------------

DYNAMIC_CELLS = [
    new_markdown_cell(
        "# High-rise 04 - Dynamic response (HFPI / SDOF)\n"
        "\n"
        "Read the Cp time series from notebook 02, build the per-floor Cf/Cm load\n"
        "history, and run the building dynamic-response recipe: generalized modal\n"
        "loads -> per-mode SDOF (RK45) integration -> per-floor displacements and\n"
        "static-equivalent loads, then off-centre accelerations for comfort. Writes\n"
        "response figures to `debug/` and a per-floor peak table to `deliverables/`.\n"
        "\n"
        "The structural model (mode shapes, floor masses, natural frequencies) comes\n"
        "from the case modes/floors/mode-shape CSVs when the `CFDMOD_HR_MODES_CSV` /\n"
        "`_FLOORS_CSV` / `_MODE_SHAPE_CSVS` variables are set; otherwise a synthetic\n"
        "cantilever sway + torsion model tuned to the case geometry is used so the\n"
        "chain runs on the fixtures."
    ),
    new_code_cell(SETUP),
    new_code_cell(
        "from cfdmod.adapters.xdmf_h5 import XdmfH5Storage\n"
        "\n"
        "# --- config -------------------------------------------------------------\n"
        "MESH = pathlib.Path(\n"
        '    os.environ.get("CFDMOD_HR_MESH", FIX / "pressure" / "galpao" / "galpao.normalized.lnas")\n'
        ")\n"
        'CASE_DATA = os.environ.get("CFDMOD_HR_CASE_DATA")\n'
        'PARAMS = os.environ.get("CFDMOD_HR_PARAMS", "params_cat3.yaml")\n'
        "if CASE_DATA:\n"
        "    case = pp.HighRiseCase.from_case_data(pathlib.Path(CASE_DATA), PARAMS)\n"
        "else:\n"
        "    case = pp.example_high_rise_case(MESH)\n"
        "\n"
        'DAMPING = float(os.environ.get("CFDMOD_HR_DAMPING", "0.02"))\n'
        "\n"
        'ARTIFACTS = OUTPUT_BASE / "artifacts" / VERSION\n'
        'cp = XdmfH5Storage(ARTIFACTS).read_data_source(pathlib.Path("cp.time_series"))\n'
        'print(f"cp: {cp.n_elements} elements, {cp.time.n_timesteps} steps | {case.n_floors} floors")'
    ),
    new_code_cell(
        "# --- per-floor load history + structural model --------------------------\n"
        'cf = pp.cf_per_floor(cp, str(MESH), case, directions=("x", "y"))\n'
        'cm = pp.cm_per_floor(cp, str(MESH), case, directions=("z",))\n'
        "load = pp.floor_load_source(cf, cm, case)\n"
        "\n"
        "# Real structural data from CSVs, or a synthetic model for the fixtures.\n"
        'MODES_CSV = os.environ.get("CFDMOD_HR_MODES_CSV")\n'
        'FLOORS_CSV = os.environ.get("CFDMOD_HR_FLOORS_CSV")\n'
        'SHAPE_CSVS = os.environ.get("CFDMOD_HR_MODE_SHAPE_CSVS")  # comma-separated, one per mode\n'
        "if MODES_CSV and FLOORS_CSV and SHAPE_CSVS:\n"
        '    structure = pp.structure_from_csvs(MODES_CSV, FLOORS_CSV, SHAPE_CSVS.split(","))\n'
        "else:\n"
        "    structure = pp.example_building_structure(case, load.n_elements)\n"
        "\n"
        "response = pp.solve_building_response(load, structure, damping_ratio=DAMPING)\n"
        "acc = pp.floor_accelerations(response, structure, point=(1.0, 0.0))\n"
        "freqs_hz = np.asarray(structure.natural_frequencies) / (2 * np.pi)\n"
        'print(f"{structure.n_modes} modes at {np.round(freqs_hz, 3)} Hz | damping {DAMPING}")'
    ),
    new_code_cell(
        "from cfdmod.dynamics import plotting as dyn\n"
        "\n"
        "# --- response figures + per-floor peak table ----------------------------\n"
        'dbg = pp.DebugWriter(OUTPUT_BASE, stage="dynamic", version=VERSION)\n'
        "\n"
        "fig, _ = dyn.plot_force_spectrum(load, freqs_hz)\n"
        'dbg.savefig(fig, "force_spectrum.png")\n'
        "plotting.close(fig)\n"
        "\n"
        "top = load.n_elements - 1\n"
        'disp_top = np.asarray(response.fields.read("disp_x"))[top]\n'
        "plim = float(1.5 * max(np.abs(disp_top).max(), 1e-9))\n"
        "fig, _ = dyn.plot_displacement(response, floor=top, plot_limit=plim)\n"
        'dbg.savefig(fig, "top_floor_hodograph.png")\n'
        "plotting.close(fig)\n"
        "\n"
        "table = pp.peak_response_table(response, acc, case)\n"
        'fig, ax = plotting.new_axes(xlabel="peak displacement [m]", ylabel="z [m]", title="Per-floor peak displacement")\n'
        'ax.plot(table["disp_x_peak"], table["z_mid"], "-o", ms=3, label="disp_x")\n'
        'ax.plot(table["disp_y_peak"], table["z_mid"], "-o", ms=3, label="disp_y")\n'
        "ax.legend()\n"
        'dbg.savefig(fig, "peak_displacement.png")\n'
        "plotting.close(fig)\n"
        "\n"
        "fig, _ = dyn.plot_acceleration_floor_by_floor(\n"
        '    table["acc_mag_peak"].to_numpy(), float(freqs_hz[0]), rec_period=10\n'
        ")\n"
        'dbg.savefig(fig, "peak_acceleration.png")\n'
        "plotting.close(fig)\n"
        "\n"
        'table.to_csv(dbg.deliverable_path("dynamic_response.csv"), index=False)\n'
        "table"
    ),
]


# --------------------------------------------------------------------------
# 05 -- facade Cp snapshots
# --------------------------------------------------------------------------

FACADE_CELLS = [
    new_markdown_cell(
        "# High-rise 05 - Facade Cp snapshots\n"
        "\n"
        "Colour the body mesh by its per-triangle Cp statistics (mean / min / max\n"
        "over the record) and render one image per facade. Facades are split by the\n"
        "outward-normal direction of each triangle (`+x` / `-x` / `+y` / `-y` sides,\n"
        "`+z` roof). Overview iso renders go to `deliverables/`; per-facade views go\n"
        "to `debug/`.\n"
        "\n"
        "Rendering uses a pure-matplotlib 3-D renderer so it runs headless. If the\n"
        "optional `[vtk]` extra (PyVista) is installed, a contoured, colour-barred\n"
        "snapshot is also written."
    ),
    new_code_cell(SETUP),
    new_code_cell(
        "from cfdmod.adapters.xdmf_h5 import XdmfH5Storage\n"
        "\n"
        "# --- config -------------------------------------------------------------\n"
        "MESH = pathlib.Path(\n"
        '    os.environ.get("CFDMOD_HR_MESH", FIX / "pressure" / "galpao" / "galpao.normalized.lnas")\n'
        ")\n"
        'ARTIFACTS = OUTPUT_BASE / "artifacts" / VERSION\n'
        'cp = XdmfH5Storage(ARTIFACTS).read_data_source(pathlib.Path("cp.time_series"))\n'
        "\n"
        "geom = pp.snapshots.load_geometry(MESH)\n"
        "n_tri = int(np.asarray(geom.triangle_vertices).shape[0])\n"
        'cp_series = np.asarray(cp.fields.read("cp"))\n'
        "cp_stats = {\n"
        '    "mean": np.nanmean(cp_series, axis=1),\n'
        '    "min": np.nanmin(cp_series, axis=1),\n'
        '    "max": np.nanmax(cp_series, axis=1),\n'
        "}\n"
        "groups = pp.snapshots.facade_groups(MESH)\n"
        'print("facades:", {k: len(v) for k, v in groups.items()})'
    ),
    new_code_cell(
        "# --- render -------------------------------------------------------------\n"
        'dbg = pp.DebugWriter(OUTPUT_BASE, stage="facade", version=VERSION)\n'
        'clim = (float(np.nanmin(cp_stats["min"])), float(np.nanmax(cp_stats["max"])))\n'
        "\n"
        "# Overview iso renders for each Cp statistic (engineer-facing).\n"
        "for stat, vals in cp_stats.items():\n"
        "    fig, _ = pp.snapshots.triangle_field_figure(\n"
        '        geom, vals, view=pp.snapshots.STANDARD_VIEWS["iso"], clim=clim,\n'
        '        title=f"{stat} Cp", cbar_label="Cp [-]",\n'
        "    )\n"
        '    dbg.savefig(fig, f"cp_{stat}_iso.png", deliverable=True)\n'
        "    plotting.close(fig)\n"
        "\n"
        "# Per-facade mean-Cp views (exploratory).\n"
        'view_for = {"n_+x": "right", "n_-x": "left", "n_+y": "front", "n_-y": "back", "n_+z": "top"}\n'
        "for name, idx in groups.items():\n"
        "    fig, _ = pp.snapshots.triangle_field_figure(\n"
        '        geom, cp_stats["mean"], subset=idx,\n'
        '        view=pp.snapshots.STANDARD_VIEWS[view_for.get(name, "iso")], clim=clim,\n'
        '        title=f"mean Cp - {pp.snapshots.FACADE_LABELS.get(name, name)}", cbar_label="Cp [-]",\n'
        "    )\n"
        '    dbg.savefig(fig, f"facade_{name}.png")\n'
        "    plotting.close(fig)\n"
        "\n"
        "# Optional high-quality PyVista render (only if the [vtk] extra is installed).\n"
        'vtp = dbg.debug_path("cp_facades.vtp")\n'
        'if pp.snapshots.write_field_vtp(geom, {"Cp_mean": cp_stats["mean"]}, vtp):\n'
        "    ok = pp.snapshots.render_vtp_snapshot(\n"
        '        vtp, dbg.deliverable_path("cp_mean_pyvista.png"),\n'
        '        scalar="Cp_mean", label="mean Cp", clim=clim,\n'
        "    )\n"
        '    print("pyvista snapshot:", ok)\n'
        "else:\n"
        '    print("pyvista/[vtk] not installed - matplotlib facade images only")\n'
        'print("facade images under", dbg.debug_dir)'
    ),
]


# --------------------------------------------------------------------------
# 06 -- structure prints
# --------------------------------------------------------------------------

STRUCTURE_CELLS = [
    new_markdown_cell(
        "# High-rise 06 - Structure prints\n"
        "\n"
        "Report-ready renders of the building model itself: plain geometry from the\n"
        "standard views, the facade partition (by outward normal), and the floor\n"
        "partition (by centroid height against the case floor edges). These document\n"
        "the model that produced the coefficients and are written to `deliverables/`\n"
        "(geometry) and `debug/` (partitions)."
    ),
    new_code_cell(SETUP),
    new_code_cell(
        "# --- config -------------------------------------------------------------\n"
        "MESH = pathlib.Path(\n"
        '    os.environ.get("CFDMOD_HR_MESH", FIX / "pressure" / "galpao" / "galpao.normalized.lnas")\n'
        ")\n"
        'CASE_DATA = os.environ.get("CFDMOD_HR_CASE_DATA")\n'
        'PARAMS = os.environ.get("CFDMOD_HR_PARAMS", "params_cat3.yaml")\n'
        "if CASE_DATA:\n"
        "    case = pp.HighRiseCase.from_case_data(pathlib.Path(CASE_DATA), PARAMS)\n"
        "else:\n"
        "    case = pp.example_high_rise_case(MESH)\n"
        "\n"
        "geom = pp.snapshots.load_geometry(MESH)\n"
        "n_tri = int(np.asarray(geom.triangle_vertices).shape[0])\n"
        "\n"
        "# Per-triangle floor index from centroid height vs the case z-edges.\n"
        "tri = np.asarray(geom.triangle_vertices)\n"
        "cz = tri[:, :, 2].mean(axis=1)\n"
        "edges = np.asarray(case.floor_heights)\n"
        "floor_id = np.clip(np.digitize(cz, edges[1:-1]), 0, len(edges) - 2).astype(float)\n"
        "groups = pp.snapshots.facade_groups(MESH)\n"
        "fac_idx = pp.snapshots.facade_index_per_triangle(groups, n_tri)\n"
        'print(f"{n_tri} triangles | {case.n_floors} floors | {len(groups)} facade groups")'
    ),
    new_code_cell(
        "# --- render -------------------------------------------------------------\n"
        'dbg = pp.DebugWriter(OUTPUT_BASE, stage="structure", version=VERSION)\n'
        "\n"
        'for v in ("iso", "front", "right", "top"):\n'
        "    fig, _ = pp.snapshots.triangle_field_figure(\n"
        '        geom, None, view=pp.snapshots.STANDARD_VIEWS[v], title=f"geometry - {v}"\n'
        "    )\n"
        '    dbg.savefig(fig, f"geometry_{v}.png", deliverable=True)\n'
        "    plotting.close(fig)\n"
        "\n"
        "fig, _ = pp.snapshots.triangle_field_figure(\n"
        '    geom, fac_idx, cmap="tab10", view=pp.snapshots.STANDARD_VIEWS["iso"],\n'
        '    title="facade partition", cbar_label="facade id",\n'
        ")\n"
        'dbg.savefig(fig, "facade_partition.png")\n'
        "plotting.close(fig)\n"
        "\n"
        "fig, _ = pp.snapshots.triangle_field_figure(\n"
        '    geom, floor_id, cmap="viridis", view=pp.snapshots.STANDARD_VIEWS["iso"],\n'
        '    title="floor partition", cbar_label="floor",\n'
        ")\n"
        'dbg.savefig(fig, "floor_partition.png")\n'
        "plotting.close(fig)\n"
        'print("structure prints under", dbg.deliverables_dir)'
    ),
]


def build() -> None:
    for fname, cells in [
        ("01_inflow.ipynb", INFLOW_CELLS),
        ("02_cp.ipynb", CP_CELLS),
        ("03_cf.ipynb", CF_CELLS),
        ("04_dynamic.ipynb", DYNAMIC_CELLS),
        ("05_facade.ipynb", FACADE_CELLS),
        ("06_structure.ipynb", STRUCTURE_CELLS),
    ]:
        nb = new_notebook(cells=cells)
        nb.metadata["kernelspec"] = {
            "display_name": "Python 3",
            "language": "python",
            "name": "python3",
        }
        nb.metadata["language_info"] = {"name": "python"}
        nbformat.write(nb, HERE / fname)
        print("wrote", HERE / fname)


if __name__ == "__main__":
    build()
