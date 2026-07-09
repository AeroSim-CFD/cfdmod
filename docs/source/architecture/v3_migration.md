# Migrating to the v3 paradigm

The v3 paradigm introduces a small set of composable primitives --
`DataSource`, `Pipeline`, `Container`, `Storage` -- and a library of
pure ops. This guide shows how to adopt them. **No legacy public symbol
has been removed.** Every function exported by `cfdmod` in v2 is still
exported in v3.

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

### Untouched legacy symbols

`run_cp`, `run_cf`, `run_cm`, `run_ce`, `apply_filters`,
`MovingAverageFilter`, `InflowData`, `Profile`, every `*Config` model,
every `LoftParams` / `RadialParams` /  `WindProfile_*` -- all unchanged.

## Side-by-side: a Cp pipeline

### Legacy (still works)

```python
from cfdmod import run_cp

run_cp(
    body_h5="body/cp.h5",
    probe_h5="probe/cp.h5",
    cfg_path="cp_config.yaml",
    output="out/",
)
```

### v3 (small-data, no I/O)

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

- New consulting notebooks: prefer v3. The data flow is explicit and
  small-data is fast.
- Existing batch scripts that hit `run_cp` / `run_cf` / etc.: keep using
  the legacy entry points. They produce the same XDMF + H5 output and
  will until v4.
- Any code that wants a custom pipeline (e.g. extra filtering step,
  alternative aggregation): build a `Pipeline` from `core_ops` rather
  than fork the legacy.

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

## When v4 lands

v4 will drop the v2 wrappers. Anything still calling `run_cp` /
`apply_filters` / `Profile.__truediv__` will need to migrate to the
recipes above. Until then, both APIs are first-class.
