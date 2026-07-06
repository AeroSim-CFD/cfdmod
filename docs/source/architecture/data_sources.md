# Data sources, ops, and pipelines (v3 paradigm)

Status: design doc for the v3 paradigm introduced by issue #131.

This document captures the locked decisions for the new abstraction layer
shipped under `cfdmod/core/` and `cfdmod/adapters/`. It is the durable
reference for "what shape is the new layer, and why". Implementation
details (file paths, signatures) live in the docstrings of those modules;
this doc is for the contract.

## 1. Goal

Replace the ad-hoc per-recipe orchestration in `cfdmod/pressure/`,
`cfdmod/inflow.py`, `cfdmod/s1/`, and `cfdmod/hfpi/` with a single
composable paradigm: a small frozen value object (`DataSource`) plus a
library of pure transformations on it.

Phases 0-2 of the plan land the abstractions and prove on-disk parity
with every existing fixture. Phases 3-8 rewrite recipe internals on top
of the new core. Existing public APIs are not removed before v4.

## 2. Data source

A `DataSource` is a frozen Pydantic value object. It holds everything
needed to describe one slab of simulation output:

- `kind`: one of `surface`, `volume`, `points`, `groups`, `modes`.
- `time`: an affine `TimeAxis(initial_time, timestep_size, n_timesteps)`.
  The full time array is *never* stored; it is reconstructed on demand
  from the three numbers. Time-aggregated outputs use `n_timesteps == 0`.
- `topology`: `Topology(cell_type, connectivity, vertices)` for surfaces
  and (later) volumes. `cell_type` is a discriminator (`triangle`,
  `point`, `cell`) so volume export is additive, not a rewrite. Points
  data sources use `cell_type="point"` and an empty connectivity array.
  `GroupsDataSource` does not own its own topology; it carries a
  *reference* to a parent surface's topology plus a per-element group
  index, so it cannot diverge from the parent mesh.
- `elements`: `ElementMeta` with positions, areas, volumes, normals as
  applicable. Optional per-element free-form metadata (e.g. station
  name) lives here.
- `groupings`: `dict[str, np.ndarray]` of int arrays of length
  `n_elements`. Each array assigns each element to exactly one group of
  the named grouping (surface name, planar/volumetric selection, S1
  separation, ...).
- `fields`: a `FieldStore`. The `FieldStore` exposes per-field arrays of
  shape `(n_elements, n_timesteps)` or `(n_elements,)` for time-aggregated
  outputs. Whether the arrays live in RAM or in an h5 file is opaque to
  the rest of the core.
- `field_meta`: `dict[str, FieldMeta]` with name, scale, units.
- `attrs`: source-level metadata (free-form, validated at construction).

The five subclasses are thin wrappers that lock the `kind` discriminator
and constrain which `ElementMeta` columns are required. They share the
same fields and methods.

## 3. Time axis is affine, not stored

The proposal is explicit about this: the time axis is reconstructable
from `(initial_time, timestep_size, n_timesteps)`. Time ops (window
selection, translation, rescale) mutate these three numbers. They never
resample. Resampling is a *field op* and goes through `FieldStore`.

## 4. Algebra: four broadcasting rules

All field algebra (Cp = `(p - p_ref) * scaling`, S1 = `profile / ref`,
etc.) goes through one module, `cfdmod/core/algebra.py`, with four
broadcasting rules dispatched on the shape pattern:

1. `[multi or single] * constant` -> uniform scaling.
2. `[multi] * [single]` with the same time axis -> column-wise (e.g.
   `p - p_ref`).
3. `[multi] * [multi]` with the same shape but only one carries
   timesteps -> row-wise (e.g. S1).
4. `[multi] * [multi]` with the same shape -> element-wise.

Recipes never reimplement broadcasting. Any new recipe (Cf, S1,
pedestrian comfort) is a `compose(...)` of existing ops + algebra calls.

## 5. Functional core, imperative shell

- Ops are pure functions: `op(ds, params) -> DataSource`. No methods on
  `DataSource` mutate state or do I/O.
- The shell (CLI, recipe runners) wires concrete adapters in via a
  frozen `Context`.
- Pipelines are `compose(*ops)` -> `Callable[[DataSource], DataSource]`,
  built once at recipe-construction time, applied at run time.

## 6. The injection seam: `FieldStore` and `Storage`

`FieldStore` is the only place the small-data (numpy in RAM) vs
large-data (XDMF + h5, 10+ GB) distinction lives. Every op calls
`fields.read(name, time_slice=..., element_slice=...)` and trusts the
adapter.

`Storage` handles whole `DataSource` round-trips: topology, time axis,
elements, groupings, plus the `FieldStore` for the data.

Two adapters land in Phase 1:

- `MemoryFieldStore` / `MemoryStorage` (in-process, dict-keyed). Used by
  every test and by notebooks. ~50 lines each.
- `H5FieldStore` / `XdmfH5Storage`. Wraps the existing
  `cfdmod.io.xdmf` writers. Preserves the on-disk layout exactly:
  `/Triangles`, `/Geometry`, `/meta/{time_steps,time_normalized,
  region_labels}`, `/{group}/t{T}`. No format change.

The `core` package depends on `numpy` and `pydantic` only; `h5py` lives
exclusively under `adapters/xdmf_h5/`.

## 7. Containers

`Container[K, V]` lifts the `HFPIAnalysisResults` pattern (hashable
Pydantic key, `join_by(callback)`, `filter_by`, `map_values`). Phase 6
aliases `HFPIAnalysisResults = Container[HFPICaseParameters, ResultType]`
so existing hfpi callers keep working.

Parallelism is *injected*, not built in: `Container.map_values(p,
pool=ctx.pool)` runs the pipeline over the container in parallel only
if a pool is supplied. Sequential is the default.

## 8. Inside vs outside the paradigm

Inside (becomes data sources + ops): `pressure/` (Cp, Cf, Cm, Ce),
`pressure/filters.py`, `inflow.py`, `s1/profile.py`, `hfpi/`, the
existing `analysis/inflow/`, `io/xdmf.py`, `io/timeseries.py`. Pedestrian
comfort joins later (it composes existing primitives + a climate-data
input).

Outside (stay standalone): `loft/`, `roughness/`, `snapshot/`,
`altimetry/`, `analytical/`, `climate/`, plotting helpers,
`io/geometry/`, `io/vtk/`. They produce inputs to or consume outputs
from data sources but never participate as filters in a pipeline.

## 9. Migration plan

Parallel API, not a hard cut. v2.x keeps every existing public function
intact. v3.x rewrites internals on the new core with the old top-level
functions as thin wrappers. v4.x drops the wrappers.

YAML schemas are not touched in phases 1-3. New schemas land under
`version: "3"`. `pressure/migrate.py` is the bridge.

## 10. What is NOT in scope

- CPU vs GPU: out of scope. If it ever matters, it is another adapter.
- `ArrayLike` protocol over `numpy.ndarray`: not the kind of swap we
  need; numpy is a hard dependency of the core.
- Abstract `BaseFilter` / `BaseRecipe` classes: composition over
  inheritance. Recipes are concrete `Pipeline`s.
- Climate, altimetry, snapshot, loft, roughness as pipeline stages:
  per the odt, these stay outside the paradigm. Pedestrian comfort
  takes climate data as a non-pipeline input.
