# cfdmod

[![Testing Pipeline](https://github.com/AeroSim-CFD/cfdmod/actions/workflows/testing.yaml/badge.svg)](https://github.com/AeroSim-CFD/cfdmod/actions/workflows/testing.yaml)
[![Docs Deploy](https://github.com/AeroSim-CFD/cfdmod/actions/workflows/pages/pages-build-deployment/badge.svg)](https://github.com/AeroSim-CFD/cfdmod/actions/workflows/pages/pages-build-deployment)
[![Linting Workflow](https://github.com/AeroSim-CFD/cfdmod/actions/workflows/linting.yaml/badge.svg)](https://github.com/AeroSim-CFD/cfdmod/actions/workflows/linting.yaml)
[![Release artifacts](https://github.com/AeroSim-CFD/cfdmod/actions/workflows/build_and_deploy_artifacts.yaml/badge.svg)](https://github.com/AeroSim-CFD/cfdmod/actions/workflows/build_and_deploy_artifacts.yaml)

Post-processing and geometry-preparation tools for CFD wind-tunnel
simulations: pressure (`Cp`), force (`Cf`), moment (`Cm`) and shape (`Ce`)
coefficients; terrain loft and roughness elements; inflow analysis;
climate / Weibull / Gumbel statistics; ParaView snapshot automation.

v3 reorganizes post-processing around a single data structure -- the
`DataSource` (elements on one axis, timesteps on the other, named fields
plus metadata) -- and composable *ops* driven by YAML **pipeline
templates**. Every flow (Cp / Cf / Cm / Ce, S1, ...) is now a template run
with `cfdmod run <template.yaml>`; the legacy per-coefficient functions
(`run_cp` / `run_cf` / ...) and their `*CaseConfig` models are gone. See
`docs/source/architecture/v3_migration.md` for the mapping and
`docs/source/architecture/data_sources.md` for the paradigm.

## Install

Base install (Cp / Cf / Cm / Ce pipeline, IO helpers, CLI):

```bash
pip install aerosim-cfdmod
```

Optional extras:

| Extras | When you need it |
|---|---|
| `[vtk]` | ParaView snapshot automation, VTK polydata writers, S1 probe-on-line |
| `[geometry]` | Altimetry section + loft helpers (trimesh) |
| `[notebook]` | jupyter / ipykernel for the worked-example notebook |
| `[docs]` | sphinx + shibuya theme + nbsphinx for `make html` |
| `[legacy]` | pandas-HDFStore compat readers (inflow, HFPI static) |

Install several at once with `pip install
"aerosim-cfdmod[vtk,geometry,notebook]"`.

`pymeshlab` is intentionally *not* an extra (its GPL license would
force GPL on downstream code). Code paths that genuinely need it
expect the user to install it explicitly at their own license risk.

## Quickstart

A pipeline is a YAML template: `inputs` (data sources on disk), a
`pipeline` of ops, and `outputs`. Run it with the CLI:

```bash
cfdmod run path/to/cp.yaml
```

A minimal Cp template -- subtract a static-pressure probe, divide by the
dynamic pressure, reduce to per-triangle statistics:

```yaml
name: cp
inputs:
  body:                          # surface pressure per triangle per timestep
    kind: surface
    path: bodies.my_case         # -> bodies.my_case.h5 (+ .xdmf)
  p_ref:                         # static reference probe. Must be named points.*
    kind: points
    path: points.static_pressure
pipeline:
  - id: cp_unscaled              # p - p_ref  (column-wise broadcast)
    kind: sub
    source: body
    rhs: p_ref
    field: pressure
    out: cp
  - id: cp_t                     # / dynamic pressure q  (scale = 1/q)
    kind: scale
    source: cp_unscaled
    field: cp
    factor: 800.0
  - id: cp_stats                 # collapse the time axis
    kind: statistics
    source: cp_t
    field: cp
    kinds: [mean, rms, min, max]
outputs:
  cp_timeseries: {source: cp_t, path: out/cp.time_series}
  cp_stats:      {source: cp_stats, path: out/cp.stats}
```

Or drive it from Python over any storage backend -- the same recipe code
runs against an in-memory store (great for notebooks / tests, no files
needed) or the on-disk XDMF+H5 store:

```python
from cfdmod import load_template, run_template, XdmfH5Storage

template = load_template("cp.yaml")
bindings = run_template(template, storage=XdmfH5Storage(root="."))
cp_t = bindings["cp_t"]          # a SurfaceDataSource; cp_t.fields.read("cp")
```

`load_template` validates the whole template up front (unknown op kinds,
dangling `source`/`rhs` refs, duplicate ids, typo'd fields) before any
file is read.

> **Filename convention:** the XDMF+H5 storage infers a source's kind from
> its filename -- a probe must be named `points.*` to load as a points
> source; everything else loads as a surface. The `kind:` you declare in
> the template is checked against the loaded kind.

Cf / Cm / Ce build on a Cp time series by attaching the mesh
(`mesh_attach`), grouping triangles (`body_grouping` / `zoning_grouping`),
and aggregating (`force_contribution` / `moment_contribution` +
`field_series_for_groups`). Complete, runnable templates for all four ship
under `fixtures/tests/pressure/templates/`.

## Examples

- **Tutorials** (`notebooks/tutorials/`): `01_data_sources` -> `02_recipes`
  -> `03_pipelines` -> `04_containers`, building from the data structure to
  full pipelines. All run on synthetic data with no fixtures.
- **Per-coefficient walkthroughs** (`docs/source/use_cases/pressure/coefficients/`):
  `calculate_{cp,Cf,Cm,Ce}.ipynb` run the shipped templates against the
  bundled `galpao` wind-tunnel fixture.
- **End-to-end** (`notebooks/process_container_pack.ipynb`): a Cp/Cf/Cm/Ce
  pipeline over a multi-container body.
- **Template reference**: `fixtures/tests/pressure/templates/{cp,cf,cm,ce}.yaml`.

## Smoothing as a pipeline step

Signal smoothing is just another op -- insert a `moving_average` step and
downstream ops consume the smoothed field. The window is in the source's
time units; no implicit unit conversion.

```yaml
  - id: cp_smoothed
    kind: moving_average
    source: cp_t
    field: cp
    window: 3.0
    out: cp
```

## CLI

```bash
cfdmod run <template.yaml>     # run a v3 pipeline template
cfdmod loft ...                # terrain loft surfaces
cfdmod roughness ...           # roughness elements (linear / radial)
cfdmod regroup ...             # split/reorder mesh triangles via a grouping chain
cfdmod altimetry ...           # altimetry section profiles
```

`python -m cfdmod <command>` works identically. `cfdmod run` prints a
one-line error (not a traceback) on a bad template and exits non-zero.

## Development

### Tests

The suite is grouped by pytest markers:

```bash
uv run pytest                          # full default suite (excludes -m perf)
uv run pytest -m unit                  # pure-function tests
uv run pytest -m integration           # end-to-end pipeline tests
uv run pytest -m perf                  # opt-in synthetic big-data benchmarks
uv run pytest tests/core tests/adapters  # the v3 data-source / storage layer
```

The perf run writes a markdown + JSON report (Python heap peak +
RSS per scale) to `output/perf/perf_report.{md,json}` so regressions
across releases are tracked over time.

### Tox

```bash
uv run tox
```

### Memory profiling

```bash
pip install -U memory-profiler
mprof run -C -M python path_to_script.py
mprof plot
```

### Schemas

Generate JSON Schema for every config model:

```bash
uv run python -m scripts.generate_schemas
```

In VSCode, point yaml.schemas at the generated file:

```json
"yaml.schemas": {
    "file:///path/to/cfdmod/output/schema-cfdmod.json": "**/cfdmod/**/*.yaml"
}
```
