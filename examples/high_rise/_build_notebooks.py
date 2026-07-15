"""Generate the high-rise stage notebooks (clean, no stored outputs).

Run: uv run python examples/high_rise/_build_notebooks.py
Writes 01_inflow, 02_cp, 03_cf, 04_dynamic next to this script.

The notebooks are thin drivers: config is read from environment variables with
in-repo fixture defaults, so they run headless (nbconvert / _validate_notebooks)
without any external data, and point at a real case by setting the CFDMOD_HR_*
variables. All reusable logic lives in the cfdmod library (cfdmod.building + cfdmod.inflow_report / report / plot_config helpers).
"""

from __future__ import annotations

import pathlib

import nbformat
from nbformat.v4 import new_code_cell, new_markdown_cell, new_notebook

HERE = pathlib.Path(__file__).resolve().parent

SETUP = """\
import os
import pathlib

import numpy as np  # noqa: F401  (used across later cells)

import matplotlib

matplotlib.use("Agg")  # headless: notebooks write files, they do not display

from cfdmod import inflow_report, plot_config  # noqa: E402
from cfdmod.building import (  # noqa: E402
    BuildingCase,
    cf_per_floor,
    cm_per_floor,
    cp_from_pressure,
    example_building_case,
    example_building_structure,
    floor_accelerations,
    floor_load_source,
    peak_response_table,
    peak_value,
    plot_floor_mass,
    plot_mode_shape,
    plot_natural_frequencies,
    solve_building_response,
    structure_from_csvs,
)
from cfdmod.report import DebugWriter  # noqa: E402


def _find_repo(start: pathlib.Path) -> pathlib.Path:
    p = start.resolve()
    while p != p.parent:
        if (p / "pyproject.toml").exists():
            return p
        p = p.parent
    return start.resolve()


REPO = _find_repo(pathlib.Path.cwd())

plot_config.apply_style()

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
        'dbg = DebugWriter(OUTPUT_BASE, stage="inflow", version=VERSION)\n'
        "inflow = InflowData.from_files(HIST, POINTS)\n"
        "profiles = inflow_report.detect_profiles(inflow, min_points=3)\n"
        'print(f"detected {len(profiles)} vertical profile(s)")\n'
        "for p in profiles:\n"
        '    print(f"  {p.name}: {len(p.point_idx)} points, z in [{p.z.min():.2f}, {p.z.max():.2f}]")'
    ),
    new_code_cell(
        "# --- per-profile figures + reference velocity ---------------------------\n"
        "u_ref_by_profile = {}\n"
        "for prof in profiles:\n"
        "    u_ref = inflow_report.reference_velocity(prof, inflow, REFERENCE_HEIGHT, component=COMPONENT)\n"
        "    u_ref_by_profile[prof.name] = u_ref\n"
        "    L = inflow_report.integral_length_scale(\n"
        "        inflow, prof.nearest_index(REFERENCE_HEIGHT), u_ref, component=COMPONENT\n"
        "    )\n"
        '    print(f"{prof.name}: u_ref(z={REFERENCE_HEIGHT:g}) = {u_ref:.4f} m/s | L = {L:.4g} m")\n'
        "\n"
        "    norm = NormalizationParameters(\n"
        "        reference_velocity=max(u_ref, 1e-9), characteristic_length=1.0\n"
        "    )\n"
        "    for name, fig in {\n"
        '        "mean_velocity": inflow_report.plot_mean_velocity(prof, inflow, component=COMPONENT),\n'
        '        "turbulence_intensity": inflow_report.plot_turbulence_intensity(\n'
        "            prof, inflow, component=COMPONENT\n"
        "        ),\n"
        '        "spectrum": inflow_report.plot_spectrum(\n'
        "            prof, inflow, REFERENCE_HEIGHT, norm, component=COMPONENT\n"
        "        ),\n"
        "    }.items():\n"
        '        dbg.savefig(fig, f"{prof.name}/{name}.png")\n'
        "        plot_config.close(fig)\n"
        'print("figures written under", dbg.debug_dir)'
    ),
    new_code_cell(
        "# --- code-standard comparison (NBR 6123 / EN 1991-1-4) ------------------\n"
        "# Overlay the simulated mean velocity / turbulence intensity on the code\n"
        "# curves, and compare the integral length scale to the EN 1991-1-4 theory.\n"
        'CAT_EU = os.environ.get("CFDMOD_HR_CAT_EU", "III")\n'
        'Z0 = float(os.environ.get("CFDMOD_HR_Z0", "0.3"))\n'
        "prof = profiles[0]\n"
        "fig, _ = inflow_report.plot_profile_vs_code(\n"
        "    prof, inflow, REFERENCE_HEIGHT, cat_eu=CAT_EU, component=COMPONENT\n"
        ")\n"
        'dbg.savefig(fig, f"{prof.name}/profile_vs_code.png", deliverable=True)\n'
        "plot_config.close(fig)\n"
        "\n"
        "L_num = inflow_report.integral_length_scale_profile(inflow, prof, component=COMPONENT)\n"
        "L_eu = inflow_report.eu_integral_length_scale(prof.z, Z0)\n"
        "fig, _ = inflow_report.plot_integral_length_scale(prof.z, L_num, REFERENCE_HEIGHT, L_theory=L_eu)\n"
        'dbg.savefig(fig, f"{prof.name}/integral_length_scale.png")\n'
        "plot_config.close(fig)\n"
        'print(f"code-comparison figures written (cat_eu={CAT_EU}, z0={Z0})")'
    ),
    new_code_cell(
        "# --- directional design speed (NBR 6123 / EN 1991-1-4) ------------------\n"
        "# Design reference speed U_H per wind direction from the wind-analysis CSVs\n"
        "# (real projects ship these under case_data; here an in-repo generic fixture).\n"
        "from cfdmod.analytical import WindProfile_EU, WindProfile_NBR\n"
        "\n"
        'V0 = float(os.environ.get("CFDMOD_HR_V0", "35.0"))\n'
        'DESIGN_HEIGHT = float(os.environ.get("CFDMOD_HR_DESIGN_HEIGHT", "100.0"))\n'
        'WIND_DIR = FIX / "inflow" / "wind_analysis"\n'
        'WIND_NBR = pathlib.Path(os.environ.get("CFDMOD_HR_WIND_NBR", WIND_DIR / "wind_analysis_NBR.csv"))\n'
        'WIND_EU = pathlib.Path(os.environ.get("CFDMOD_HR_WIND_EU", WIND_DIR / "wind_analysis_EU.csv"))\n'
        "\n"
        "u_h_nbr = inflow_report.directional_reference_speed(\n"
        "    WindProfile_NBR.build(WIND_NBR, V0=V0), height=DESIGN_HEIGHT, recurrence_period=50, use_kd=True\n"
        ")\n"
        "u_h_eu = inflow_report.directional_reference_speed(\n"
        "    WindProfile_EU.build(WIND_EU, Vb=V0), height=DESIGN_HEIGHT, recurrence_period=50, use_kd=True\n"
        ")\n"
        'print(f"governing U_H @ {DESIGN_HEIGHT:g} m: '
        'NBR {u_h_nbr.max():.2f} @ {u_h_nbr.idxmax():g} deg | EU {u_h_eu.max():.2f} @ {u_h_eu.idxmax():g} deg")\n'
        "\n"
        "fig, ax = plot_config.new_axes(\n"
        '    xlabel="wind direction [deg]", ylabel="U_H [m/s]", title="Directional design speed"\n'
        ")\n"
        'ax.plot(u_h_nbr.index, u_h_nbr.to_numpy(), "-o", ms=3, label="NBR 6123")\n'
        'ax.plot(u_h_eu.index, u_h_eu.to_numpy(), "-s", ms=3, label="EN 1991-1-4")\n'
        "ax.legend()\n"
        'dbg.savefig(fig, "directional_U_H.png", deliverable=True)\n'
        "plot_config.close(fig)\n"
        "\n"
        'table = u_h_nbr.rename("U_H_NBR").to_frame().join(u_h_eu.rename("U_H_EU"))\n'
        'dbg.save_csv(table.rename_axis("wind_direction").reset_index(), "directional_U_H.csv", deliverable=True)'
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
        "    case = BuildingCase.from_case_data(pathlib.Path(CASE_DATA), PARAMS)\n"
        "else:\n"
        "    case = example_building_case(MESH)\n"
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
        "cp = cp_from_pressure(body, p_ref, case)\n"
        'cp_stats = cp_from_pressure(body, p_ref, case, statistics=["mean", "rms", "min", "max"])\n'
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
        'dbg = DebugWriter(OUTPUT_BASE, stage="cp", version=VERSION)\n'
        'fig, ax = plot_config.new_axes(xlabel="mean Cp [-]", ylabel="count", title="Cp mean distribution")\n'
        "ax.hist(mean_cp[np.isfinite(mean_cp)], bins=40)\n"
        'dbg.savefig(fig, "cp_mean_hist.png")\n'
        "plot_config.close(fig)\n"
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
        "    case = BuildingCase.from_case_data(pathlib.Path(CASE_DATA), PARAMS)\n"
        "else:\n"
        "    case = example_building_case(MESH)\n"
        "\n"
        'ARTIFACTS = OUTPUT_BASE / "artifacts" / VERSION\n'
        'cp = XdmfH5Storage(ARTIFACTS).read_data_source(pathlib.Path("cp.time_series"))\n'
        'print(f"cp: {cp.n_elements} elements, {cp.time.n_timesteps} steps | {case.n_floors} floors")'
    ),
    new_code_cell(
        "# --- per-floor Cf / Cm --------------------------------------------------\n"
        'cf = cf_per_floor(cp, str(MESH), case, directions=("x", "y"))\n'
        'cm = cm_per_floor(cp, str(MESH), case, directions=("z",))\n'
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
        'dbg = DebugWriter(OUTPUT_BASE, stage="cf", version=VERSION)\n'
        'fig, ax = plot_config.new_axes(xlabel="mean coefficient [-]", ylabel="z [m]", title="Per-floor Cf / Cm")\n'
        'ax.plot(cfx_mean, z_mid, "-o", ms=3, label="Cf_x")\n'
        'ax.plot(cfy_mean, z_mid, "-o", ms=3, label="Cf_y")\n'
        'ax.plot(cmz_mean, z_mid, "-s", ms=3, label="Cm_z")\n'
        "ax.legend()\n"
        'dbg.savefig(fig, "per_floor_coefficients.png")\n'
        "plot_config.close(fig)\n"
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
        "    case = BuildingCase.from_case_data(pathlib.Path(CASE_DATA), PARAMS)\n"
        "else:\n"
        "    case = example_building_case(MESH)\n"
        "\n"
        'DAMPING = float(os.environ.get("CFDMOD_HR_DAMPING", "0.02"))\n'
        "\n"
        'ARTIFACTS = OUTPUT_BASE / "artifacts" / VERSION\n'
        'cp = XdmfH5Storage(ARTIFACTS).read_data_source(pathlib.Path("cp.time_series"))\n'
        'print(f"cp: {cp.n_elements} elements, {cp.time.n_timesteps} steps | {case.n_floors} floors")'
    ),
    new_code_cell(
        "# --- per-floor load history + structural model --------------------------\n"
        'cf = cf_per_floor(cp, str(MESH), case, directions=("x", "y"))\n'
        'cm = cm_per_floor(cp, str(MESH), case, directions=("z",))\n'
        "load = floor_load_source(cf, cm, case)\n"
        "\n"
        "# Real structural data from CSVs, or a synthetic model for the fixtures.\n"
        'MODES_CSV = os.environ.get("CFDMOD_HR_MODES_CSV")\n'
        'FLOORS_CSV = os.environ.get("CFDMOD_HR_FLOORS_CSV")\n'
        'SHAPE_CSVS = os.environ.get("CFDMOD_HR_MODE_SHAPE_CSVS")  # comma-separated, one per mode\n'
        "if MODES_CSV and FLOORS_CSV and SHAPE_CSVS:\n"
        '    structure = structure_from_csvs(MODES_CSV, FLOORS_CSV, SHAPE_CSVS.split(","))\n'
        "else:\n"
        "    structure = example_building_structure(case, load.n_elements)\n"
        "\n"
        "response = solve_building_response(load, structure, damping_ratio=DAMPING)\n"
        "acc = floor_accelerations(response, structure, point=(1.0, 0.0))\n"
        "freqs_hz = np.asarray(structure.natural_frequencies) / (2 * np.pi)\n"
        'print(f"{structure.n_modes} modes at {np.round(freqs_hz, 3)} Hz | damping {DAMPING}")'
    ),
    new_code_cell(
        "from cfdmod.dynamics import plotting as dyn\n"
        "\n"
        "# --- response figures + per-floor peak table ----------------------------\n"
        'dbg = DebugWriter(OUTPUT_BASE, stage="dynamic", version=VERSION)\n'
        "\n"
        "fig, _ = dyn.plot_force_spectrum(load, freqs_hz)\n"
        'dbg.savefig(fig, "force_spectrum.png")\n'
        "plot_config.close(fig)\n"
        "\n"
        "top = load.n_elements - 1\n"
        'disp_top = np.asarray(response.fields.read("disp_x"))[top]\n'
        "plim = float(1.5 * max(np.abs(disp_top).max(), 1e-9))\n"
        "fig, _ = dyn.plot_displacement(response, floor=top, plot_limit=plim)\n"
        'dbg.savefig(fig, "top_floor_hodograph.png")\n'
        "plot_config.close(fig)\n"
        "\n"
        "table = peak_response_table(response, acc, case)\n"
        'fig, ax = plot_config.new_axes(xlabel="peak displacement [m]", ylabel="z [m]", title="Per-floor peak displacement")\n'
        'ax.plot(table["disp_x_peak"], table["z_mid"], "-o", ms=3, label="disp_x")\n'
        'ax.plot(table["disp_y_peak"], table["z_mid"], "-o", ms=3, label="disp_y")\n'
        "ax.legend()\n"
        'dbg.savefig(fig, "peak_displacement.png")\n'
        "plot_config.close(fig)\n"
        "\n"
        "fig, _ = dyn.plot_acceleration_floor_by_floor(\n"
        '    table["acc_mag_peak"].to_numpy(), float(freqs_hz[0]), rec_period=10\n'
        ")\n"
        'dbg.savefig(fig, "peak_acceleration.png")\n'
        "plot_config.close(fig)\n"
        "\n"
        'dbg.save_csv(table, "dynamic_response.csv", deliverable=True)\n'
        "table"
    ),
    new_code_cell(
        "# --- structural model figures + peak-acceleration methods ---------------\n"
        "# Visualise the structural model that drove the response ...\n"
        "fig, _ = plot_mode_shape(structure, 0, rotation_scale=structure.floors_radius)\n"
        'dbg.savefig(fig, "mode_0.png")\n'
        "plot_config.close(fig)\n"
        "fig, _ = plot_floor_mass(structure)\n"
        'dbg.savefig(fig, "floor_mass.png")\n'
        "plot_config.close(fig)\n"
        "fig, _ = plot_natural_frequencies(structure)\n"
        'dbg.savefig(fig, "natural_frequencies.png")\n'
        "plot_config.close(fig)\n"
        "\n"
        "# ... and compare peak-estimation methods on the top-floor acceleration.\n"
        'acc_top = np.asarray(acc.fields.read("acc_mag"))[top]\n'
        "peak_methods = {\n"
        "    m: peak_value(acc_top, m, f0=float(freqs_hz[0]))\n"
        '    for m in ("max", "peak-factor", "gumbel")\n'
        "}\n"
        'print("top-floor peak acc [m/s^2]:", {k: round(v, 4) for k, v in peak_methods.items()})'
    ),
]


def build() -> None:
    for fname, cells in [
        ("01_inflow.ipynb", INFLOW_CELLS),
        ("02_cp.ipynb", CP_CELLS),
        ("03_cf.ipynb", CF_CELLS),
        ("04_dynamic.ipynb", DYNAMIC_CELLS),
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
