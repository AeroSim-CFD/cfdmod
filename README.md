# cfdmod

[![Testing Pipeline](https://github.com/AeroSim-CFD/cfdmod/actions/workflows/testing.yaml/badge.svg)](https://github.com/AeroSim-CFD/cfdmod/actions/workflows/testing.yaml)
[![Docs Deploy](https://github.com/AeroSim-CFD/cfdmod/actions/workflows/pages/pages-build-deployment/badge.svg)](https://github.com/AeroSim-CFD/cfdmod/actions/workflows/pages/pages-build-deployment)
[![Linting Workflow](https://github.com/AeroSim-CFD/cfdmod/actions/workflows/linting.yaml/badge.svg)](https://github.com/AeroSim-CFD/cfdmod/actions/workflows/linting.yaml)
[![Release artifacts](https://github.com/AeroSim-CFD/cfdmod/actions/workflows/build_and_deploy_artifacts.yaml/badge.svg)](https://github.com/AeroSim-CFD/cfdmod/actions/workflows/build_and_deploy_artifacts.yaml)

Post-processing and geometry-preparation tools for CFD wind-tunnel
simulations: pressure (`Cp`), force (`Cf`), moment (`Cm`) and shape (`Ce`)
coefficients; terrain loft and roughness elements; inflow analysis;
climate / Weibull / Gumbel statistics; ParaView snapshot automation.

v2.0 redesigned the pressure pipeline around external consumers: an XDMF+H5
output contract end-to-end, no input mutation, embedded post-processing
metadata, multi-format geometry (`.lnas` / `.stl` / `.h5` / `.xdmf`), and
flat output by default. See `docs/source/release_notes.md` for the full
v2.0 changeset.

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
| `[legacy]` | pandas-HDFStore compat readers (inflow, HFPI static, migrate) |

Install several at once with `pip install
"aerosim-cfdmod[vtk,geometry,notebook]"`.

`pymeshlab` is intentionally *not* an extra (its GPL license would
force GPL on downstream code). Code paths that genuinely need it
expect the user to install it explicitly at their own license risk.

## Quickstart

Install the package and run the pipeline against an existing body + probe
XDMF+H5 pair (the layout produced by the AeroSim CFD solver):

```python
from cfdmod import (
    BasicStatisticModel, BodyConfig, BodyDefinition, CpCaseConfig, CpConfig,
    CfCaseConfig, CfConfig, CmCaseConfig, CmConfig, MomentBodyConfig,
    ZoningModel, run_cp, run_cf, run_cm,
)
from cfdmod.io.geometry.transformation_config import TransformationConfig

cp_cfg = CpCaseConfig(
    pressure_coefficient={
        "default": CpConfig(
            statistics=[BasicStatisticModel(stats="mean")],
            timestep_range=(150.0, 260.0),
            simul_U_H=1.0,
            simul_characteristic_length=10.0,
            # Optional: defaults are 'pressure' and 'probe' respectively.
            # macroscopic_type: 'pressure' | 'rho'
            # reference_pressure: 'probe' (point above body) | 'average'
        )
    }
)

# 1) Cp from body + probe; geometry is read from body_h5 by default. Pass
#    mesh_path= to embed a single fixed-frame reference mesh into the cp_h5
#    output (useful when running several wind directions whose body H5s are
#    rotated copies of each other) -- run_cf / run_cm inherit it from cp_h5.
run_cp(
    body_h5="body.h5",
    probe_h5="probe.h5",
    cfg_path=cp_cfg,        # in-memory config -- YAML path also accepted
    output="output",
    # mesh_path optional; .lnas / .stl / .h5 / .xdmf all supported
)

# 2) Cf from the cp.time_series.h5 produced above. nominal_area is required:
#    cfdmod will not pick a tribute area for you (so the resulting Cf can be
#    converted back to Forces unambiguously).
run_cf(
    cp_h5="output/cp.default.time_series.h5",
    cfg_path=CfCaseConfig(
        bodies={"my_body": BodyDefinition(surfaces=[])},  # [] = whole mesh
        force_coefficient={
            "scan": CfConfig(
                statistics=[BasicStatisticModel(stats="mean")],
                bodies=[BodyConfig(name="my_body", sub_bodies=ZoningModel())],
                directions=["x", "y", "z"],
                nominal_area=100.0,  # m^2 -- e.g. building frontal area
                transformation=TransformationConfig(),
            )
        },
    ),
    output="output",
)

# 3) Cm with per-region overturning moments about each container's footprint
#    base. lever_strategy="region_bbox_corners_xy" expands every body into
#    four independent runs (xmin_ymin, xmin_ymax, xmax_ymin, xmax_ymax).
#    nominal_volume is required (same rationale as nominal_area for Cf).
run_cm(
    cp_h5="output/cp.default.time_series.h5",
    cfg_path=CmCaseConfig(
        bodies={"my_body": BodyDefinition(surfaces=[])},
        moment_coefficient={
            "scan": CmConfig(
                statistics=[BasicStatisticModel(stats="mean")],
                bodies=[
                    MomentBodyConfig(
                        name="my_body",
                        sub_bodies=ZoningModel(),
                        lever_strategy="region_bbox_corners_xy",
                    )
                ],
                directions=["x", "y", "z"],
                nominal_volume=1000.0,  # m^3 -- e.g. building bounding-box volume
                transformation=TransformationConfig(),
            )
        },
    ),
    output="output",
)
```

`output/` afterwards contains, flat:

| File | Contents |
|---|---|
| `cp.{label}.time_series.{h5,xdmf}` | Cp animation on the full mesh |
| `Cf.{label}.{body}.time_series.{h5,xdmf}` | Cf animation per body (3 directional Attributes per timestep) |
| `Cm.{label}.{body}[.{case}].time_series.{h5,xdmf}` | Cm animation per body / case |
| `Ce.{label}.time_series.{h5,xdmf}` | Ce animation on the sliced regions mesh |
| `Ce.{label}.regions.stl` | Cut regions mesh as STL |
| `stats.{h5,xdmf}` | Combined statistics with one Grid per leaf group |

Every output H5 carries the post-processing config under
`/processing_metadata/`; read it back with
`cfdmod.io.read_processing_metadata(path, group)`.

## Filtering between coefficients

Filters are an opt-in pipeline step: take any `*.time_series.h5`,
apply a chain in order, and write a new `*.time_series.h5` with the
applied chain recorded under `/processing_metadata`. Cf / Cm / Ce
then consume the filtered file in place of the raw Cp.

```python
from cfdmod import MovingAverageFilter, apply_filters

apply_filters(
    input_h5="output/cp.default.time_series.h5",
    output_h5="output/cp.default.smoothed.time_series.h5",
    filters=[MovingAverageFilter(window=3.0)],   # in input time units
    group="cp",
)
# Then point run_cf / run_cm at the smoothed file.
```

`MovingAverageFilter.window` is in the same units as the input file's
time axis (raw solver time when `CpConfig.normalize_time=False`, the
default; convective time when `True`) -- the filter performs no
implicit unit conversion.

## Worked example notebook

`notebooks/process_container_pack.ipynb` runs the full Cp/Cf/Cm pipeline on
a multi-container body, auto-detects container partition from triangle
centroids using a >1 m gap rule, and produces a four-corner overturning
moment scan per container -- without authoring any surfaces or sub-bodies.
Drop a body H5 and a probe H5 next to the repo root and execute the
notebook top-to-bottom.

## CLI

The same pipeline is reachable via `python -m cfdmod pressure` (or `cfdmod
pressure` after install):

```bash
python -m cfdmod pressure cp \
    --body body.h5 --probe probe.h5 \
    --config cp.yaml --output output

python -m cfdmod pressure cm \
    --cp output/cp.default.time_series.h5 \
    --config cm.yaml --output output
```

`--mesh` is optional; it accepts `.lnas` / `.stl` / `.h5` / `.xdmf` and
defaults to the geometry embedded in `--body` / `--cp`.

## Development

### Tests

The suite is grouped by pytest markers:

```bash
uv run pytest                          # full default suite (excludes -m perf)
uv run pytest -m unit                  # pure-function tests
uv run pytest -m integration           # end-to-end pipeline tests
uv run pytest -m perf                  # opt-in synthetic big-data benchmarks
uv run pytest tests/io tests/pressure  # the v2 pipeline scope
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
