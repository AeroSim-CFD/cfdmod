# Tests

The suite is organised into three categories selected by pytest markers,
plus per-module fixtures factored out into `conftest.py` files. The
default `pytest` invocation runs everything except `perf`; the perf
benchmark is opt-in.

## Categories

| Marker        | What it covers                                                    | Typical wall time |
|---------------|-------------------------------------------------------------------|--------------------|
| `unit`        | Pure-function tests with no real-data fixtures or pipeline calls. | < 10 s total       |
| `integration` | End-to-end pipeline runs that read real fixture H5 files.         | < 30 s total       |
| `perf`        | Synthetic-data benchmarks of the full Cp/Cf/Cm chain.             | seconds-minutes    |

`perf` tests are excluded from the default run via `addopts = "-m 'not perf'"`
in `pyproject.toml`.

## Running selectively

```bash
pytest                           # default: 'not perf'
pytest -m unit                   # fast, ~10 s
pytest -m integration            # ~30 s
pytest -m "unit or integration"  # same as default for the marked tests
pytest -m perf                   # opt-in, synthetic data
```

## Perf scales

The `perf` module synthesises body+probe XDMF+H5 in a fixture, sized via
`CFDMOD_PERF_SCALE`:

| Scale                | Triangles  | Timesteps  | Purpose                              |
|----------------------|-----------:|-----------:|--------------------------------------|
| `tiny`               |      5 000 |        200 | quick smoke (~3 s total)             |
| `medium` *(default)* |     30 000 |      2 000 | standard benchmark (~minute total)   |
| `extreme`            |    150 000 |     10 000 | 1/5x of worst observed real case     |

```bash
CFDMOD_PERF_SCALE=tiny    pytest -m perf -s   # quick check
CFDMOD_PERF_SCALE=extreme pytest -m perf -s   # the long one
```

Per-stage budgets default per scale; override individually via
`CFDMOD_PERF_{CP,CF,CM}_BUDGET_S` and `CFDMOD_PERF_RSS_BUDGET_MB`.
The `_measure()` context manager prints a one-line `[perf:{scale}]
{label}: ...` report so `pytest -m perf -s` doubles as an interactive
benchmark.

### Per-run report

Every `pytest -m perf` invocation also writes a structured report:

```
output/perf/perf_report.md     # human-readable
output/perf/perf_report.json   # programmatic
```

The report path is configurable via `CFDMOD_PERF_REPORT_DIR` (default
`output/perf`). Both files are overwritten per run; check them in
manually if you want to keep history.

For each pipeline stage the report captures:

- `elapsed_s` -- wall time.
- `rss_after_mib` -- process-wide peak RSS so far (includes numpy /
  h5py native allocations).
- `rss_delta_mib` -- the rise during the stage; lower bound on additional
  peak observed.
- `py_heap_peak_mib` -- peak Python-managed heap during the stage
  (`tracemalloc`); useful for catching accidental in-Python copies.

Sample output (medium scale, this dev machine):

| Stage | Wall (s) | RSS after | RSS delta | Py heap peak |
|---|---:|---:|---:|---:|
| synthesise_inputs | 1.66 | 347 | 13 | 3 |
| run_cp | 4.39 | 441 | 94 | 42 |
| run_cf | 20.03 | 884 | 443 | 467 |
| run_cm (4 cases) | 90.40 | 1021 | 138 | 510 |

## Shared fixtures (`tests/pressure/conftest.py`)

The pressure suite shares its boilerplate via a `conftest.py`:

- **Path constants**: `BUILDING_BODY_H5`, `BUILDING_PROBE_H5`,
  `GALPAO_BODY_H5`, `GALPAO_PROBE_H5`, `GALPAO_MESH`,
  `BIG_BODY_H5`, `BIG_PROBE_H5`.
- **Helpers**: `zoning_full()`, `basic_stats(*names)`,
  `iter_stats_leaves(h5_path)`.
- **Config builders**: `make_cp_cfg`, `make_cf_cfg`, `make_cm_cfg` --
  return a `CaseConfig` instance with sensible test defaults and
  `**kwargs` overrides.
- **Session-scoped fixtures**: `building_cp_h5`, `galpao_cp_h5` run Cp
  once per pytest session and reuse the resulting timeseries H5 across
  every downstream test in the same session.

## Categorisation at a glance

| Path                                          | Category    |
|-----------------------------------------------|-------------|
| `tests/io/test_xdmf.py`                       | unit        |
| `tests/io/test_mesh.py`                       | unit        |
| `tests/io/test_region_meshing.py`             | unit        |
| `tests/io/test_STL.py`                        | unit        |
| `tests/io/test_write_vtk.py`                  | unit        |
| `tests/io/test_probe_vtm.py`                  | integration |
| `tests/pressure/test_extreme_values.py`       | unit        |
| `tests/pressure/test_functions_ce.py`         | unit        |
| `tests/pressure/test_functions_cf.py`         | unit        |
| `tests/pressure/test_functions_cm.py`         | unit        |
| `tests/pressure/test_functions_zoning.py`     | unit        |
| `tests/pressure/test_geometry.py`             | unit        |
| `tests/pressure/test_statistics.py`           | unit        |
| `tests/pressure/test_functions_cp.py`         | integration |
| `tests/pressure/test_migrate.py`              | integration |
| `tests/pressure/test_run_cm_cases.py`         | integration |
| `tests/pressure/test_run_multi_body.py`       | integration |
| `tests/pressure/test_statistics_runner.py`    | integration |
| `tests/pressure/test_perf.py`                 | perf        |
| `tests/analysis/inflow/test_functions.py`     | integration |
| `tests/altimetry/test_altimetry.py`           | unit        |
| `tests/config/test_hashable.py`               | unit        |
| `tests/test_utils.py`                         | unit        |
| `tests/loft/`, `tests/roughness/`, `tests/s1/` | unmarked (pre-v2 modules) |
