# CLAUDE.md - cfdmod project guide

## What this project is

`aerosim-cfdmod` is a Python library for post-processing and geometry preparation of CFD wind tunnel simulations. It covers:

- Terrain loft surface generation (circular projection to a fixed radius)
- Roughness element placement (linear grid and radial ring patterns)
- Pressure coefficient analysis (Cp, Cf, Cm, Ce)
- Inflow profile analysis
- Climate/wind statistics (Weibull, Gumbel, Lawson comfort)
- Altimetry section analysis
- ParaView/VTK snapshot automation

---

## Conventions

- **No Unicode or special characters** in any file (notebooks, scripts, YAML, markdown).
  Use plain ASCII only: `->` not arrows, `x` not multiplication sign, `u_mean` not Greek letters, `^2` not superscripts.
  For equations in legends, feel free to use Latex notation for equations and special symbols in Latex representation.
- **No inline comments in python3 -c "..." terminal commands** (shell escaping issues).
- **NEVER reference internal GitHub issues or PRs in public documentation.** This is a
  HARD RULE. Anything under `docs/`, `README.md`, tutorial/notebook READMEs, docstrings,
  or any other user-facing published text must NOT contain issue/PR numbers, `#<n>`
  references, `gh issue`/`github.com/.../issues` links, or phrasings like
  "(issue #131)". Issue and PR references are internal-only (commit messages, PR
  descriptions, and issue comments). Release notes describe *what changed* for users, not
  which issue tracked it - do not cite issue numbers there either. When documenting a
  feature, describe the behavior, not the work item that produced it.
- Python >= 3.10; formatting, import-sorting, and linting all via ruff (`ruff format` + `ruff check`).
- Configuration via YAML + Pydantic v2 `BaseModel` with `from_file(path)` classmethods.
- External mesh format: `aerosim-lnas` (`LnasFormat`, `LnasGeometry`). Prefer lnas over trimesh for loading STL/LNAS surfaces.
- Testing: pytest, fixture files under `fixtures/tests/`, tests mirror source structure under `tests/`.
- **Plans for issues belong on the issue.** When you produce an implementation plan tied to an existing GitHub issue, post it as a comment on that issue (`gh issue comment <n>`) so the design lives next to the work item.
- **Implement on a feature branch, not on main.** Before starting any non-trivial implementation, create a branch (`git checkout -b feat/<short-name>` or `fix/<short-name>`) off main and commit work there in logical chunks. Never accumulate uncommitted work on main.
- **Always push commits.** After committing on a feature branch, push it to `origin` (`git push`) so the remote branch / PR stays current. Do not leave commits sitting only in the local worktree.

---

## Architecture (v3 paradigm, issue #131)

The library is a **pure functional core**: one immutable value object
(`DataSource`) transformed by pure functions (`ops`), composed into
`Pipeline`s either programmatically (`recipes`) or declaratively (YAML
templates). I/O lives entirely behind `Protocol` seams in `adapters/`.
See `docs/source/architecture/data_sources.md` and `v3_migration.md`.

Public symbols import from the top-level package:

```python
from cfdmod import DataSource, SurfaceDataSource, Pipeline, compose, Container
from cfdmod import MemoryStorage, XdmfH5Storage, load_template, run_template
from cfdmod.recipes import build_cp, cf_pipeline, cm_pipeline, ce_pipeline
from cfdmod.core.ops.field import moving_average  # ops layer
```

### Directory layout

```
cfdmod/
    cfdmod/
        __init__.py         Lazy public API (PEP-562 __getattr__)
        __main__.py         Global typer CLI (loft|roughness|regroup|altimetry|run)
        core/               v3 paradigm:
            data_source.py  DataSource + 5 kinds (Surface/Volume/Points/Groups/Modes)
            ops/            pure ops: time / field / geometric / data_source_create
            recipes/        Cp, Cf, Cm, Ce, s1, dynamic, pedestrian_comfort (compose ops)
            pipeline.py     compose(*ops); pipeline_yaml.py: YAML templates + OP_REGISTRY
            container.py    Container[K,V] for multi-case fan-out
        adapters/           storage seam: memory/ (tests) and xdmf_h5/ (production)
        io/                 geometry (STL/lnas load), vtk (ParaView probe/write)
        inflow.py           InflowData + inflow analysis functions (single file)
        inflow_report.py    ABL profile detection + inflow-validation figures
        hfpi/               legacy HFPI dynamic pipeline (RK45 SDOF, reporting)
        geometry/grouping/  triangle-index grouping specs (By*Grouping union)
        regroup/            disk regroup: new lnas + reordered h5 timeseries
        remesh/             QEM decimation per group
        building/           building wind-load post-pro (BuildingCase, per-floor Cf/Cm, dynamic response)
        report.py           DebugWriter: versioned debug/ + deliverables/ output roots
        mesh_field.py       per-triangle mesh-field renders (matplotlib; optional PyVista .vtp)
        plot_config.py      shared matplotlib style helpers (apply_style/new_axes/close)
        loft/ roughness/ altimetry/ climate/ analytical/ s1/ snapshot/
        logger.py  utils.py
    tests/                  Mirror of cfdmod/ structure (pytest markers: unit/integration/perf)
    fixtures/tests/         YAML configs + STL/LNAS + h5 fixtures (galpao, caarc, inflow, ...)
    notebooks/              tutorials/ (v3 API teaching notebooks)
    examples/               use-case suites (high_rise, container_pack, roughness, s1_topographic)
    docs/                   Sphinx documentation
```

Note: `cfdmod/api/` and `cfdmod/analysis/inflow/` hold only stale `.pyc`
cruft (not git-tracked); ignore them. There is no `cfdmod/pressure/`
package -- Cp/Cf/Cm/Ce are recipes under `core/recipes/`.

### Layout inside each domain module

```
module/
    __init__.py     Exports all public symbols for the module
    __main__.py     python -m entry point -> calls cli.app()
    cli.py          Thin typer app (uses run.py functions)
    run.py          Pure Python orchestration (no argparse, no file paths)
    parameters.py   Pydantic BaseModel config models with a from_file classmethod
    functions.py    Core computational logic
    [helpers].py    Supporting modules
```

Data flow:
```
YAML file -> Pydantic model (parameters.py)
          -> run() orchestration (run.py)
          -> core functions (functions.py)
          -> output objects (returned to caller or written by cli.py)
```

### CLI entry points

Each module is runnable as a module and as a subcommand of the global CLI:

```bash
python -m cfdmod.loft --config ... --surface ... --output ...
python -m cfdmod.roughness --config ... --output ... --mode radial

python -m cfdmod loft --config ... --surface ... --output ...
python -m cfdmod roughness --config ... --output ...
```

After install, the `cfdmod` shell command is also available (via `[project.scripts]`).

### Config models

All config classes are Pydantic v2 ``BaseModel`` subclasses (or
``BasePressureConfig`` for the pressure-coefficient configs, which itself
extends ``BaseModel``). The standard surface:

```python
params = LoftParams.from_file("config.yaml")   # YAML -> instance (per-class classmethod)
params = LoftParams(**{...})                    # kwargs / dict
params.model_dump()                             # -> dict
params.model_dump_json()                        # -> JSON string
```

There is no project-specific config base class; field declarations follow the
``Annotated[T, Field(...)]`` form from Pydantic v2.

---

## Notebook utilities

`cfdmod.notebook_utils` provides lightweight helpers for exploratory work:

- `mesh_summary(path)` - print triangle/vertex counts and bounding box for .lnas or .stl
- `show_config(config)` - pretty-print any Pydantic ``BaseModel`` config as a dict
- `load_lnas(path)` - load an .lnas file and return the LnasFormat object

All three are also exported from the top-level `cfdmod` package.

---

## Post-processing notebook suite (`examples/high_rise/`)

Application-directed post-processing lives in `examples/`, built on the v3
recipes/ops. The **high-rise** suite is the reference layout:

- **Thin notebooks, one per stage.** Notebooks orchestrate; they hold no
  reusable logic. All reusable logic lives in the cfdmod library: `cfdmod.building`
  (`BuildingCase` case_data aggregation, `cp_from_pressure`, per-floor Cf/Cm,
  dynamic response), `cfdmod.report.DebugWriter` (output roots),
  `cfdmod.inflow_report` (ABL validation), `cfdmod.mesh_field` (mesh renders),
  `cfdmod.plot_config` (figure style). Nothing high-rise-specific is siloed --
  the same helpers serve low-rise and other building studies.
- **Output, not inline results.** Notebooks write images/tables to versioned
  roots instead of storing results inline:
  `<case>/debug/<version>/<stage>/...` (free-to-compare exploratory output) and
  `<case>/deliverables/<version>/<stage>/...` (engineer-facing). Re-running the
  same `version` overwrites in place; a new `version` coexists.
- **High-rise sequence:** inflow validation (extract U_H at reference height)
  -> update case dynamic pressure -> Cp -> per-floor Cf/Cm -> dynamic analysis
  -> deliverables + verbose debug -> facade Cp snapshots.
- Cf/Cm use **explicit reference-area** normalisation (`nominal_area` /
  `nominal_volume`), not the legacy per-region bounding-box area.

`examples/high_rise/_validate_high_rise.py` exercises the `cfdmod.building`
helpers end-to-end on the galpao / pitot_inlet fixtures
(`uv run python examples/high_rise/_validate_high_rise.py`).

---

