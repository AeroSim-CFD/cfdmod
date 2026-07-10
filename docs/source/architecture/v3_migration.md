# Migrating to the v3 paradigm

The v3 paradigm introduces a small set of composable primitives --
`DataSource`, `Pipeline`, `Container`, `Storage` -- and a library of
pure ops. This guide shows how to adopt them. The disk-first v2 pressure
entry points have been removed: post-processing now runs through the v3
recipes and the `cfdmod run <template.yaml>` CLI.

## What changed at the import surface

### New top-level symbols

```python
from cfdmod import (
    # value objects
    DataSource, SurfaceDataSource, VolumeDataSource,
    PointsDataSource, GroupsDataSource, ModesDataSource,
    TimeAxis, Topology, ElementMeta, Grouping, FieldMeta,
    # composition
    Pipeline, compose, Container,
    # adapters (storage seam)
    MemoryStorage, MemoryFieldStore,
    XdmfH5Storage, H5FieldStore,
    # ops + recipes namespaces
    core_ops, recipes,
)
```

### Removed and relocated symbols

The `cfdmod.pressure` package has been removed. `run_cp`, `run_cf`,
`run_cm`, `run_ce`, `apply_filters`, and `MovingAverageFilter` no longer
exist; use the v3 recipes (`build_cp`, `cf_pipeline`, `cm_pipeline`,
`ce_pipeline`) or a YAML template run with `cfdmod run <template.yaml>`.
The functional filtering that `apply_filters` / `MovingAverageFilter`
provided is now the `moving_average` field op, composable into any
pipeline.

Everything else is unchanged: `InflowData`, `Profile`, every `*Config`
model, and every `LoftParams` / `RadialParams` / `WindProfile_*` are
still exported from `cfdmod`.

## A Cp pipeline (small-data, no I/O)

```python
import numpy as np
from cfdmod import (
    SurfaceDataSource, TimeAxis, Topology, ElementMeta,
    MemoryFieldStore,
)
from cfdmod.recipes import CpRecipeConfig, build_cp

body = SurfaceDataSource(
    time=TimeAxis(initial_time=0.0, timestep_size=0.001, n_timesteps=10_000),
    topology=Topology.triangles(connectivity, vertices),
    elements=ElementMeta(area=area_array),
    fields=MemoryFieldStore({"pressure": pressure_array}),
)
cp = build_cp(
    body,
    p_ref=ref_data_source_or_scalar,
    cfg=CpRecipeConfig(dynamic_pressure=0.5 * 1.225 * 12.0**2,
                       statistics=["mean", "rms", "peak_max"]),
)
# cp is a SurfaceDataSource with time-aggregated stat fields.
```

The same recipe runs against `XdmfH5Storage` for production-size data
without changing the pipeline -- only the `Storage` adapter swaps.

## When to reach for v3

- New consulting notebooks: prefer the recipes. The data flow is
  explicit and small-data is fast.
- Batch post-processing: author a YAML template and run it with
  `cfdmod run <template.yaml>`. The template declares its own inputs,
  pipeline steps, and outputs, and produces the XDMF + H5 outputs via
  `XdmfH5Storage`.
- Any code that wants a custom pipeline (e.g. extra filtering step,
  alternative aggregation): build a `Pipeline` from `core_ops`.

## Recipe reference

| Recipe | Inputs | Output | Module |
|---|---|---|---|
| `build_cp` | body + reference data sources | surface (with stats) | `cfdmod.recipes.cp` |
| `cf_pipeline` | grouped Cp data source | groups (per-body Cf) | `cfdmod.recipes.cf` |
| `cm_pipeline` | grouped moment-contribution data source | groups (per-body Cm) | `cfdmod.recipes.cm` |
| `ce_pipeline` | zoned Cp data source | groups (per-zone Ce) | `cfdmod.recipes.ce` |
| `build_s1` | CFD profile + reference profile | points (S1 vs height) | `cfdmod.recipes.s1` |
| `build_pedestrian_comfort` | velocity field + probes | points (probe stats) | `cfdmod.recipes.pedestrian_comfort` |
| `build_dynamic_response` | force field + mode shapes + solver | points (response) | `cfdmod.recipes.dynamic` |

## Op reference

Every recipe is `compose(...)` of these ops; you can build your own.

| Family | Ops |
|---|---|
| Time | `window_selection`, `translate`, `rescale` |
| Field | `add`, `sub`, `mul`, `div`, `scale`, `moving_average` |
| Geometric | `attach_grouping` |
| Source-create | `compute_statistics`, `field_series_for_groups`, `filter_by_grouping`, `probe_extraction`, `profile_interpolation`, `modal_projection`, `modal_recomposition` |

All ops are pure functions: `op(ds, params) -> DataSource`.

## Where the old entry points went

The v2 pressure entry points (`run_cp`, `apply_filters`, and the
`cfdmod.pressure` package internals) have been removed. Migrate to the
recipes above, or express the workflow as a YAML template and run it
with `cfdmod run <template.yaml>`. The filtering step that
`apply_filters` / `MovingAverageFilter` provided is now the
`moving_average` field op.
