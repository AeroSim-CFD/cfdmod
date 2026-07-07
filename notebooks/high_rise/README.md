# High-rise post-processing suite (v3)

Thin, application-directed notebooks for high-rise wind-load post-processing,
built on the cfdmod v3 recipes/ops. Each notebook is one stage of the sequence
and holds no reusable logic -- the shared glue lives in `pp/` (a notebook-side
helper package, deliberately NOT part of the cfdmod library).

## Sequence

| Notebook | Stage | In | Out |
|---|---|---|---|
| `01_inflow.ipynb` | Inlet profile validation | probe hist series + points | mean/TI/spectrum figs to `debug/`; `u_ref` at reference height to `deliverables/` |
| `02_cp.ipynb` | Pressure coefficient | body + reference pressure | `cp.time_series` to `artifacts/`; stats to `deliverables/` |
| `03_cf.ipynb` | Per-floor Cf / Cm | `cp.time_series` + mesh | per-floor coefficient figs to `debug/`; load table to `deliverables/` |

Dynamic analysis, deliverables/debug reporting, and facade snapshots are the
next stages (Phase 3+), not yet in this suite.

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
uv run python notebooks/high_rise/_validate_pp.py         # unit-level checks on the pp/ helpers
uv run python notebooks/high_rise/_validate_notebooks.py  # execute 01->02->03 on fixtures
```

Point at a real case with environment variables (or by editing the config cell):

| Variable | Meaning | Default |
|---|---|---|
| `CFDMOD_HR_OUTPUT_BASE` | output root | `notebooks/high_rise/_run` |
| `CFDMOD_HR_VERSION` | output version tag | `example` |
| `CFDMOD_HR_INFLOW_HIST` / `_POINTS` | inflow data | pitot_inlet fixture |
| `CFDMOD_HR_REF_HEIGHT` | reference height H | `2.0` |
| `CFDMOD_HR_CASE_DATA` / `CFDMOD_HR_PARAMS` | case_data dir + params yaml | example case from mesh |
| `CFDMOD_HR_DATA_DIR` / `_BODY_KEY` / `_REF_KEY` | pressure data | galpao fixture |
| `CFDMOD_HR_MESH` | body `.lnas` | galpao normalized |

## Helper package (`pp/`)

- `HighRiseCase` -- aggregate `case_data` (global_data.json + params yaml);
  derive dynamic pressure; `with_reference_velocity(u_ref)`.
- `example_high_rise_case(mesh)` -- self-contained case for demos/tests.
- `DebugWriter` -- versioned `debug/` + `deliverables/` paths.
- `inflow_report` -- vertical-profile detection + validation figures.
- `pressure` -- `cp_from_pressure`, `cf_per_floor`, `cm_per_floor` (explicit
  reference-area normalisation).

The `_*.py` files are dev tooling (generate / validate the notebooks), not part
of the suite itself. Regenerate the notebooks after editing them with
`uv run python notebooks/high_rise/_build_notebooks.py`.

## Notes / open items

- Cf/Cm use **explicit reference-area** normalisation (`nominal_area` /
  `nominal_volume`), per the 3.2 decision -- not the legacy per-region
  bounding-box area, so values differ from the 067 deliverables.
- Numerical validation currently runs on the in-repo `galpao` / `caarc` /
  `pitot_inlet` fixtures. Case 067 has config but no raw pressure data on disk;
  case 063 (Codeme) has real high-rise raw data for a real-case run.
- A YAML-template batch path (`cfdmod run`) exists for Cp/Cf/Cm/Ce, but the
  per-floor Cf template with dynamic dynamic-pressure is a follow-up (templates
  currently bake a fixed scale factor).
