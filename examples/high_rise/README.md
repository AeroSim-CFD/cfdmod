# High-rise post-processing suite (v3)

Thin, application-directed notebooks for high-rise wind-load post-processing,
built on the cfdmod v3 recipes/ops. Each notebook is one stage of the sequence
and holds no reusable logic -- the shared glue lives in the cfdmod library:
`cfdmod.building` (case + per-floor Cf/Cm + dynamic response), plus
`cfdmod.inflow_report`, `cfdmod.mesh_field`, `cfdmod.report`, `cfdmod.plot_config`.

## Sequence

| Notebook | Stage | In | Out |
|---|---|---|---|
| `01_inflow.ipynb` | Inlet profile validation | probe hist series + points | mean/TI/spectrum figs to `debug/`; `u_ref` at reference height to `deliverables/` |
| `02_cp.ipynb` | Pressure coefficient | body + reference pressure | `cp.time_series` to `artifacts/`; stats to `deliverables/` |
| `03_cf.ipynb` | Per-floor Cf / Cm | `cp.time_series` + mesh | per-floor coefficient figs to `debug/`; load table to `deliverables/` |
| `04_dynamic.ipynb` | Dynamic response (HFPI / SDOF) | `cp.time_series` + mesh + structural model | response figs to `debug/`; per-floor peak table to `deliverables/` |
| `05_facade.ipynb` | Facade Cp snapshots | `cp.time_series` + mesh | per-facade mean-Cp views to `debug/`; overview iso renders to `deliverables/` |
| `06_structure.ipynb` | Structure prints | mesh | facade/floor partitions to `debug/`; geometry views to `deliverables/` |

## Output layout

Notebooks do not store results inline. They write to versioned roots under an
`OUTPUT_BASE`:

```
<OUTPUT_BASE>/debug/<version>/<stage>/...         # free-to-compare exploratory
<OUTPUT_BASE>/deliverables/<version>/<stage>/...  # engineer-facing
<OUTPUT_BASE>/artifacts/<version>/...             # intermediate data (e.g. cp.time_series)
```

Re-running the same `version` overwrites in place; a new `version` coexists.

## Running

Every notebook reads its config from environment variables with in-repo fixture
defaults, so the whole chain runs headless with no external data:

```bash
uv run python examples/high_rise/_validate_high_rise.py  # unit-level checks on the cfdmod.building helpers
uv run python examples/high_rise/_validate_notebooks.py  # execute 01->06 on fixtures
```

Point at a real case with environment variables (or by editing the config cell):

| Variable | Meaning | Default |
|---|---|---|
| `CFDMOD_HR_OUTPUT_BASE` | output root | `examples/high_rise/_run` |
| `CFDMOD_HR_VERSION` | output version tag | `example` |
| `CFDMOD_HR_INFLOW_HIST` / `_POINTS` | inflow data | pitot_inlet fixture |
| `CFDMOD_HR_REF_HEIGHT` | reference height H | `2.0` |
| `CFDMOD_HR_CAT_EU` / `CFDMOD_HR_Z0` | terrain category + roughness for the code comparison (stage 01) | `III` / `0.3` |
| `CFDMOD_HR_WIND_NBR` / `_WIND_EU` | directional wind-analysis CSVs (stage 01) | `inflow/wind_analysis/` fixture |
| `CFDMOD_HR_V0` / `CFDMOD_HR_DESIGN_HEIGHT` | basic wind speed + design height for the directional U_H (stage 01) | `35.0` / `100.0` |
| `CFDMOD_HR_CASE_DATA` / `CFDMOD_HR_PARAMS` | case_data dir + params yaml | example case from mesh |
| `CFDMOD_HR_DATA_DIR` / `_BODY_KEY` / `_REF_KEY` | pressure data | galpao fixture |
| `CFDMOD_HR_MESH` | body `.lnas` | galpao normalized |
| `CFDMOD_HR_DAMPING` | modal damping ratio (stage 04) | `0.02` |
| `CFDMOD_HR_MODES_CSV` / `_FLOORS_CSV` / `_MODE_SHAPE_CSVS` | structural model CSVs (stage 04) | synthetic model |

## Library helpers used by the notebooks

All reusable logic lives in the cfdmod library (nothing is high-rise-specific):

- `cfdmod.building` -- `BuildingCase` (aggregate `case_data`; derive dynamic
  pressure; `with_reference_velocity(u_ref)`), `example_building_case(mesh)`,
  `cp_from_pressure`, `cf_per_floor` / `cm_per_floor` (explicit reference-area
  normalisation), and the dynamic-response wiring (`floor_load_source`,
  `example_building_structure` / `structure_from_csvs`, `solve_building_response`,
  `floor_accelerations`, `peak_response_table`).
- `cfdmod.report.DebugWriter` -- versioned `debug/` + `deliverables/` paths.
- `cfdmod.inflow_report` -- vertical-profile detection + validation figures.
- `cfdmod.mesh_field` -- `facade_groups`, `triangle_field_figure`,
  `facade_index_per_triangle`, optional PyVista `write_field_vtp` /
  `render_vtp_snapshot`.
- `cfdmod.plot_config` -- shared figure style (`apply_style`, `new_axes`, `close`).

The `_*.py` files are dev tooling (generate / validate the notebooks), not part
of the suite itself. Regenerate the notebooks after editing them with
`uv run python examples/high_rise/_build_notebooks.py`.

## Notes / open items

- Cf/Cm use **explicit reference-area** normalisation (`nominal_area` /
  `nominal_volume`), per the 3.2 decision -- not the legacy per-region
  bounding-box area, so values differ from the earlier per-region deliverables.
- Numerical validation currently runs on the in-repo `galpao` / `caarc` /
  `pitot_inlet` fixtures. One real consulting case has config but no raw
  pressure data on disk; another has real high-rise raw data for an
  end-to-end run.
- A YAML-template batch path (`cfdmod run`) exists for Cp/Cf/Cm/Ce, but the
  per-floor Cf template with dynamic dynamic-pressure is a follow-up (templates
  currently bake a fixed scale factor).
- Stage 04 uses a **synthetic cantilever structural model** (sway + torsion
  modes, Ellis `46/H` fundamental) on the fixtures; point it at a real modal
  model with the `CFDMOD_HR_MODES_CSV` / `_FLOORS_CSV` / `_MODE_SHAPE_CSVS`
  variables (columns per `cfdmod.dynamics.structural`).
- Stages 05/06 render with a pure-matplotlib 3-D renderer so they run headless.
  Installing the optional `[vtk]` extra (PyVista) additionally writes a
  contoured, colour-barred facade snapshot; without it, only the matplotlib
  images are produced.
