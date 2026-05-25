# Release Notes

## v2.1.0

Minor release on top of v2.0.1. Three geometry-preprocessing
subpackages land -- the triangle-grouping pipeline
(`cfdmod.geometry.grouping`), the regroup module (`cfdmod.regroup`),
and the remesh module (`cfdmod.remesh`) -- all exported from the
top-level `cfdmod` package and driven from the new notebooks. The
release also relaxes the dependency version bounds so `aerosim-cfdmod`
can be co-installed as a library without forcing resolution conflicts
on consumers. No breaking changes to existing public APIs; the legacy
`sub_bodies` YAML form and the v2.0 Cp/Cf/Cm/Ce output layout are
unchanged.

### Triangle-grouping pipeline (`cfdmod.geometry.grouping`)

A new top-level subpackage promotes triangle grouping to a
first-class pipeline step, mirroring the v2.0 filter pipeline for
time-series. Specs are Pydantic models in a discriminated union
(`GroupingSpec`), composed left-to-right with `apply_groupings`, and
a triangle may belong to zero, one, or many groups. Three kinds ship
out of the box:

- `BySurfaceGrouping` -- collect named LNAS surfaces into one or
  more groups (generalises the legacy `BodyDefinition.surfaces` and
  `CeConfig.sets`).
- `ByZoningGrouping` -- axis-aligned centroid binning into a
  Cartesian grid of regions (generalises `ZoningModel`).
- `ByDivisionsGrouping` -- convenience wrapper over `ByZoningGrouping`
  that splits each axis into `n_div_{x,y,z}` equal cells over the
  candidate centroid bounding box.
- `BySizeGrouping` -- convenience wrapper that builds cells of a
  fixed `size_{x,y,z}` per axis anchored at the bounding-box minimum.
- `ByConnectivityGrouping` -- one group per connected component of
  the (sub)mesh, defined by shared-edge adjacency. The first kind
  the legacy `sub_bodies` field could not express.
- `ByNormalGrouping` -- bucket triangles by the cardinal direction
  their outward normal best aligns with (windward / leeward / roof /
  sidewall in one shot, with a tolerance angle).
- `ByPlaneGrouping` -- bin centroids by signed distance from an
  oriented plane. Generalises `ByZoningGrouping` to non-axis-aligned
  half-space splits; the default produces two halves on either side
  of the plane.
- `ByPercentileGrouping` -- equal-count quantile bins along one axis,
  for stable per-cell statistics when triangle density is uneven.
- `ByCylindricalGrouping` -- Cartesian product of (r, theta, axial)
  bins around a user-chosen cylinder axis. Natural fit for towers,
  silos, chimneys and any partition that prefers cylindrical
  coordinates over Cartesian.
- `CustomGrouping` -- escape hatch for grouping logic the built-in
  kinds cannot express: user supplies a Python callback (importable
  dotted path or a direct callable) plus an opaque `params` dict, and
  the callback returns the standard `{group_name: triangle_indices}`
  mapping. Callback signature is fixed (`(mesh, candidate_idxs,
  params) -> dict`); the driver validates indices are in range and
  honor `restrict_to`. Dotted-path callbacks round-trip cleanly
  through `dump_groupings` / `load_groupings`; direct callables are
  serialised by deriving their import path when stable (lambdas and
  local functions raise on serialise).

Each spec exposes a `restrict_to: list[str] | None` field so later
steps can scope their work to triangles already in earlier groups;
this is how the legacy `surface -> sub_body` nesting is reproduced.

`BodyConfig` (and `MomentBodyConfig`) gain an opt-in
`groupings: list[GroupingSpec] | None` field. When set, it replaces
the implicit `[BySurface, ByZoning(sub_bodies)]` chain entirely;
mixing it with a non-default `sub_bodies` is rejected. The legacy
YAML form (`sub_bodies` only) keeps working untouched -- the
canonical chain is synthesized internally via
`BodyConfig.resolved_groupings(sfc_list)`.

Internally, `cfdmod/pressure/geometry.py` was refactored to consume
`GroupingResult` end-to-end. `GeometryData.zoning_to_use` is gone;
the new fields are `grouping`, `spec_chain`, and `body_label`.
Region labels (`"{idx}-{body}"` and the `"-1-{body}"` sentinel for
unbinned triangles) are byte-identical with the legacy output, so
existing downstream consumers, notebooks, and `process_container_pack`
runs are unaffected.

`CeConfig.zoning` keeps its current per-surface, normal-axis-aware
shape; promoting it onto the same explicit-chain pattern is tracked
as a follow-up.

A grouping chain may be persisted alongside the existing `filters`
chain in `processing_metadata`:

```python
write_processing_metadata(h5, "/", {
    "filters": [spec.model_dump() for spec in filter_chain],
    "groupings": dump_groupings(grouping_chain),
})
```

See `fixtures/tests/pressure/Cf_params_groupings.yaml` for the new
explicit YAML form, and the API reference for the full surface.

### Regroup module (`cfdmod.regroup`)

A standalone preprocessing module that takes a geometry plus a
per-triangle HDF5 timeseries (Cp-style: rows = timesteps, columns =
parent triangle ids), applies a chain of triangle-grouping specs
(Ce-style 90-degree zoning cuts, connectivity-based container
separation, etc.), and writes two aligned outputs:

- a new `LnasFormat` mesh with one named surface per group, and
- a new HDF5 timeseries whose columns line up with the new triangle
  order (`per_triangle` mode) or whose values are area-weighted
  aggregates per group broadcast over each group's triangles
  (`area_weighted_mean` mode).

It reuses `cfdmod.geometry.grouping` for all binning kinds and adds
one regroup-local spec, `BySizeRoundedPerComponent`, that fans out
per-component target-size subdivisions (resolved by `run_regroup`).
The module follows the standard domain layout (config / cli / run /
functions) and is runnable as `python -m cfdmod.regroup`. New public
symbols exported from the top-level package: `RegroupConfig`,
`RegroupSpec`, `RegroupIndex`, `BySizeRoundedPerComponent`,
`build_regroup_mapping`, `build_regrouped_mesh`,
`apply_regroup_to_timeseries`, `expand_regroup_chain`, `run_regroup`.
Driven end-to-end from the new `notebooks/regroup_containers.ipynb`,
which folds slice + per-face aggregation + remesh into a single
in-memory pipeline.

### Remesh module (`cfdmod.remesh`)

Per-group triangle coarsening for grouped `LnasFormat` meshes. The
default path is exact coplanar-fan merging: within each group,
adjacent triangles that share a plane are replaced by the minimum
triangulation of their joined region (a flat NxN-subdivided square
collapses to 2 triangles). It is lossless, deterministic, and pulls
in no external dependency. An opt-in path runs Quadric Error Metrics
(QEM) decimation on top via `fast-simplification` (MIT-licensed,
available through the new `remesh` extra) for groups on curved
surfaces where coplanar merge cannot reduce the count further.

API surface only -- no CLI, no YAML config, no HDF5 I/O -- intended
for in-process use from notebooks and debugging scripts. New public
symbols exported from the top-level package: `merge_coplanar`,
`decimate_qem`, `remesh_per_group`.

### Relaxed dependency bounds

The runtime dependency pins were loosened from capped ranges to
floors-only so `aerosim-cfdmod` can be co-installed as a library
without forcing version-resolution conflicts on downstream projects
(for example one that needs pandas 3.x). The upper caps were dropped
from `numpy`, `scipy`, `pandas`, `pydantic`, `ruamel-yaml`,
`colorama`, `matplotlib`, `pyarrow`, and `aerosim-lnas`, and from the
`geometry`, `remesh`, and `vtk` optional extras. Lower bounds (the
minimum tested versions) are unchanged. The `legacy`/`dev` `tables`
pin keeps its `<4` cap, which guards Python 3.10 support. Consuming
applications should pin exact versions themselves; this library only
declares floors.

## v2.0.1

Patch release on top of v2.0.0 covering the worked-notebook polish
and one Cm regression that surfaced the first time a user ran
`run_cm` with the new reference-mesh path on a real container pack.

### Notebook (`process_container_pack.ipynb`)

- The geometry-inspect and zoning auto-detect cells now load
  `REFERENCE_MESH` when set, falling back to the body H5's embedded
  geometry when not. Previously they always called
  `mesh_from_h5(BODY_H5)`, so when a user pointed `REFERENCE_MESH`
  at a fixed-frame STL the auto-detect partitioned the *body H5's*
  wind-aligned frame and then applied those boundaries to the
  reference-frame Cp output -- producing wrong region splits along
  whichever axis the wind was aligned with.
- Zoning detection is now **hierarchical 2D**. The previous
  implementation projected centroids on each axis independently,
  which lumped misaligned containers together along whichever axis
  had row overlap. New algorithm: pick the xy axis with the largest
  single gap as the *primary* split, then within each primary
  subset detect secondary boundaries on the other xy axis; the
  Cartesian `ZoningModel`'s secondary intervals are the union of
  all per-subset boundaries. Misaligned rows now produce extra
  empty cells (harmless -- the pipeline treats them as zero-area
  regions) but each container ends up in its own region.
- Z is no longer partitioned automatically; `z_intervals` is
  fixed at `[-inf, +inf]`. The container-pack use case has all
  containers on the ground; vertical splits are out of scope.
- The auto-detect cell now prints the resulting `x_intervals`,
  `y_intervals`, `z_intervals` so users can map regions to
  physical containers without inspecting the model object.

### Pressure pipeline

- `_resolve_region_origin` (in `cfdmod/pressure/functions.py`)
  used to parse the integer prefix of a region label via
  `int(region_label.split("-", 1)[0])`. For triangles whose
  centroid did not fall into any zoning cell --
  `cfdmod.pressure.geometry.get_indexing_mask` initialises the
  region array to `-1` and only updates entries that match a
  cell -- the resulting label was `"-1-<body>"`, and the split
  produced an empty-string head, raising
  `ValueError: invalid literal for int() with base 10: ''`. Cf
  and Ce survived because they only group by the string label;
  only Cm parsed the integer back. Switched the parser to
  `rsplit("-", 1)` so the body suffix is peeled from the right
  and negative ints survive: `"-1-pack"` -> `("-1", "pack")` ->
  `int("-1")`. Sentinel triangles now fall through to the
  `lever_strategy` branch (region_base computes a meaningful
  `(mean_x, mean_y, min_z)` from their own rows; fixed picks up
  the body's `lever_origin`). Three regression tests under
  `tests/pressure/test_functions_cm.py` pin the behaviour for
  fixed, region_base, and explicit-override on a negative key.

## v2.0.0

API-first rewrite of the post-processing pipeline. The branch focus was
"library that external scripts and notebooks can drive" -- the public API,
the I/O contract and the output layout were all redesigned around that goal.

### Sane defaults and explicit knobs

The Cp config inputs were tightened so the pipeline never silently
guesses something the user didn't ask for. All of these are
behaviour-changing relative to v1.x:

- `CpConfig.macroscopic_type` now defaults to `"pressure"` (was
  `"rho"`). The description lists both options. If the solver wrote
  real pressure -- the common case -- you can leave it unset.
- New `CpConfig.reference_pressure: Literal["probe", "average"]`,
  default `"probe"`. `"probe"` uses the first probe point (the
  reference probe placed above the body, the standard wind-tunnel
  choice); `"average"` takes the spatial mean across all probe points
  at each timestep.
- New `CpConfig.normalize_time: bool = False`. Time-axis
  normalisation is now opt-in: with the default `False`,
  `/meta/time_normalized` carries raw solver time (nothing is
  silently divided by `L/U`). Filters and statistics downstream
  operate in whichever units this setting selects.
  `simul_characteristic_length` becomes meaningful only when
  `normalize_time=True`; `simul_U_H` stays a hard requirement (it is
  in the Cp dynamic-pressure denominator regardless).
- `CfConfig.nominal_area` and `CmConfig.nominal_volume` are now
  required (`gt=0`). The previous "fall back to tribute area / volume
  when zero" behaviour is gone -- without an explicit reference
  value, the resulting Cf / Cm cannot be converted back to real-scale
  forces / moments unambiguously, so the program no longer chooses
  for you. The unreachable tribute-area / tribute-volume code paths
  in `transform_Cf` / `transform_Cm` were removed.
- For Cf / Cm, `full_scale_U_H` and `full_scale_characteristic_length`
  on `ExtremeGumbelParamsModel` are now optional. When omitted the
  runner reads them from the Cp metadata embedded in `cp_h5`
  (`/processing_metadata`) -- so you only need to specify those
  scales once, in the Cp scope. Cp itself still requires explicit
  values.

### Reference-frame override (multi-direction sweeps)

`run_cp(mesh_path=...)` now actually embeds that mesh's triangles +
vertices into the Cp output via `process_xdmf_to_cp(mesh_override=...)`,
with a triangle-count safety check. Use case: same building, several
wind directions; each solver run produces a body H5 in its own wind-
aligned ("spun") coordinate frame, and you want all cp / cf / cm / ce
outputs in a single fixed reference frame for cross-direction
comparison. Downstream `run_cf` / `run_cm` / `run_ce` already default
mesh from `cp_h5`, so the reference frame propagates without re-passing
`mesh_path` per call.

### First-class filter chain

Signal-processing filters are now their own pipeline stage rather than
being smuggled into the statistics block:

- `cfdmod.pressure.filters.apply_filters(input_h5, output_h5,
  filters=[...], group=...)` reads any coefficient timeseries, applies
  the chain in order, and writes a new `*.time_series.h5` with the
  same on-disk shape (`/Triangles + /Geometry`, `/{group}/t{T}` per
  timestep, `/meta/...`, sibling temporal XDMF). The applied chain is
  recorded under `/processing_metadata` so the lineage is self-
  describing.
- Initial filter type: `MovingAverageFilter(window=...)`. `window` is
  in the input file's own time-axis units (raw solver time by
  default; convective time when `normalize_time=True`). No implicit
  unit conversion. Implemented via a flat Pydantic discriminated
  union, so a new filter is one new class added to the union and one
  branch in `_apply_one`.
- `ExtremeMovingAverageParamsModel`,
  `moving_average_extreme_values`, and the `"Moving Average"` entry
  in `ExtremeMethods` were **removed**. Statistics now expose only
  the three real peak-factor methods: `Absolute`, `Peak`, `Gumbel`.
  Moving-average smoothing is done in the filter stage, then
  statistics run over the filtered file.

### Pipeline (Cp / Cf / Cm / Ce)

- Disk-first stats contract. Every coefficient persists its full per-triangle
  timeseries to an XDMF+H5 file *before* statistics are computed. Statistics
  are then read back from disk via
  `cfdmod.pressure.statistics_runner.calculate_statistics_from_h5` so memory
  pressure no longer scales with the number of timesteps.
- Single combined `stats.{h5,xdmf}` for the whole run, with an embedded mesh
  per leaf group. `write_stats_xdmf` walks the H5 tree and emits one
  `<Grid>` per `(coefficient, body[, direction[, case]])` triple, each on
  the correct sub-mesh -- fixes the silent length-mismatch behaviour in
  v1.x where Cf/Cm/Ce stats were written against the full-mesh topology
  while their values were per-body or per-region.
- Multi-attribute temporal XDMF for Cf/Cm body timeseries: pick `cf_x`,
  `cf_y`, `cf_z` (or `cm_x/y/z`) from the ParaView Attribute selector on the
  same animation.
- The user's input H5 files are read-only. The previous in-place mutator
  (`add_cp2xdmf`) is gone, and a regression test pins body / probe size and
  modification-time across a full `run_cp`.
- Multi-format mesh resolver. `mesh_path` now accepts `.lnas` / `.stl` /
  `.h5` / `.xdmf` (or a pre-loaded `LnasFormat`). It is also optional --
  when omitted, the geometry is read from the source H5's embedded
  `/Triangles + /Geometry`. Internally LnasFormat is still the carrier;
  externally STL/XDMF/H5 are first-class.
- Embedded post-processing metadata. Every output H5 carries
  `<group>/processing_metadata/config.yaml` plus group attrs for
  `produced_at`, `cfdmod_version`, `coefficient`, `cfg_lbl`, `body`,
  `direction`, and the input paths. `read_processing_metadata(path, group)`
  round-trips back to a dict.
- Output layout is flat by default: every artefact for a `(coefficient,
  cfg_lbl[, body[, case]])` triple sits directly in `output_path` with
  dot-separated filenames (`cp.default.time_series.h5`,
  `Cf.containers.pack.time_series.h5`, ...). Combined stats land in
  `stats.{h5,xdmf}`.

### New Cm features

- `lever_strategy="region_base"` derives a per-region base from each
  region's triangle vertices `(mean_x, mean_y, min_z)` -- the natural
  reference for overturning moments about the floor footprint.
- `lever_strategy="region_bbox_corners_xy"` expands one body into four
  independent runs (`xmin_ymin`, `xmin_ymax`, `xmax_ymin`, `xmax_ymax`)
  so external pipelines can scan worst-case overturning moments around
  every footprint corner without doing the orchestration themselves.
- `region_lever_origins: dict[int, (x, y, z)]` for explicit per-region
  centers (HFPI-style externally-known centers of mass).
- `lever_origin_cases: dict[case_label, dict[region_int, (x, y, z)]]` for
  arbitrary case scans.

### Public API

- `from cfdmod import run_cp, run_cf, run_cm, run_ce` -- canonical pipeline
  entry points.
- `from cfdmod import load_mesh, mesh_from_h5,
  read_processing_metadata, write_processing_metadata` -- IO helpers
  surfaced for external consumers.
- `from cfdmod import read_timeseries_df, to_csv, plot_timeseries` --
  pull a `pd.DataFrame` out of any `*.time_series.h5` (with optional
  triangle / region / timestep filtering), export to CSV, or plot
  selected columns with one call. `regions=True` deduplicates the
  per-triangle broadcast of Cf/Cm so you get one column per region
  instead of one per triangle.
- `from cfdmod import MovingAverageFilter, apply_filters` -- the
  filter chain (see "First-class filter chain" above). `FilterSpec`
  is also exported for type annotations of user-built chains.
- `cfg_path` accepts either a YAML path *or* a pre-built
  `CpCaseConfig` / `CfCaseConfig` / `CmCaseConfig` / `CeCaseConfig`
  instance, so in-memory pipelines don't need sidecar YAMLs.
- CLI subcommands (`python -m cfdmod pressure cp|cf|cm|ce`) accept all
  mesh formats with `--mesh` now optional.

### Breaking changes

- `cfdmod.pressure.add_cp2xdmf` removed (it mutated the input body H5).
  External callers should use `run_cp`, which writes to a separate
  output file.
- `cfdmod.pressure.add_lever_arm_to_geometry_df` signature changed:
  the third argument is now a `MomentBodyConfig` instead of a bare
  `lever_origin` tuple, so the per-region lever logic can be applied
  without the caller re-implementing it.
- `MomentBodyConfig.lever_origin` is now optional with default
  `(0.0, 0.0, 0.0)`. Configs that previously relied on it being required
  will continue to load.
- `cfdmod.pressure.path_manager.{get_results_h5_path, get_results_xdmf_path}`
  renamed to `get_stats_h5_path` / `get_stats_xdmf_path`. The output files
  are now `stats.{h5,xdmf}` (was `results.{h5,xdmf}`).
- `cfdmod.api` and `cfdmod.use_cases` shims have been **removed**.
  v1.x scripts that imported via these paths must update to the
  top-level domain modules (`cfdmod.io` and the per-domain packages
  such as `cfdmod.loft`, `cfdmod.pressure`, `cfdmod.roughness`, ...).
- `cfdmod.config` and `cfdmod.HashableConfig` have been **removed**.
  The base class added a `to_yaml()` method that no caller used, a
  `to_dict()` method that was a one-line wrapper around Pydantic's
  built-in `model_dump()`, and a `sha256()` config-fingerprint helper
  that only one scratch notebook ever called. Configs now subclass
  `pydantic.BaseModel` directly. Migration:
    - `config.to_dict()` -> `config.model_dump()`
    - `config.to_yaml(path)` -> serialise via your YAML library of
      choice (e.g. `ruamel.yaml.YAML().dump(config.model_dump(), fh)`)
    - `config.sha256()` -> compute it externally:
      `hashlib.sha256(config.model_dump_json().encode()).hexdigest()`
  The per-`*CaseConfig` `from_file(path)` classmethods are unchanged.
- Pre-typer argparse entry points were removed: `cfdmod.loft.main`,
  `cfdmod.roughness.main`, and `cfdmod.altimetry.main`. Each module
  now exposes a typer app at `cfdmod.<module>.cli:app`, registered
  under the unified `python -m cfdmod <module>` entry point.
- `cfdmod.snapshot.__main__` was removed (it imported a non-existent
  `cfdmod.snapshot.main`). The snapshot module currently has no CLI
  entry point; use the Python API directly until one lands.
- `CpConfig.macroscopic_type` default flipped from `"rho"` to
  `"pressure"`. Configs that rely on the implicit default but actually
  feed LBM density now need to set `macroscopic_type="rho"` explicitly.
- `CpConfig` time-axis normalisation is now opt-in via
  `normalize_time: bool = False`. The previous behaviour (always
  divide by `L/U`) becomes `normalize_time=True`. With the new
  default, `/meta/time_normalized` carries raw solver time, and
  filter / Gumbel windows operate in the same raw-time units.
- `CfConfig.nominal_area` and `CmConfig.nominal_volume` are now
  required (`gt=0`); the implicit-tribute fallback was removed.
  Configs that previously left them unset (or set them to `0`) need
  to provide an explicit reference area / volume. Only `transform_Cf`
  / `transform_Cm` were affected; the public `get_representative_areas`
  / `get_representative_volume` helpers are still exported.
- `ExtremeMovingAverageParamsModel`,
  `cfdmod.pressure.functions.moving_average_extreme_values`, and the
  `"Moving Average"` entry in `ExtremeMethods` were removed. To get
  stats over a moving-average-smoothed signal, run
  `apply_filters([MovingAverageFilter(window=...)])` first and then
  run statistics over the filtered file.
- The pressure pipeline is a single chain now: `run_cp` -> optional
  `apply_filters` -> `run_cf` / `run_cm` / `run_ce` -> stats merged
  into `stats.{h5,xdmf}`. The in-memory transform helpers
  (`process_xdmf_to_cp`, `process_Cf`, `process_Cm`, `process_Ce`,
  `process_timeseries`, `process_surfaces`, `transform_Cf`,
  `transform_Cm`, `transform_Ce`, `calculate_statistics`,
  `tabulate_geometry_data`, `combine_stats_data_with_mesh`,
  `add_lever_arm_to_geometry_df`, `get_indexing_mask`,
  `get_representative_areas`, `get_representative_volume`,
  `get_surface_dict`) and the data-class containers
  (`GeometryData`, `ProcessedEntity`, `CommonOutput`, `CeOutput`)
  are no longer part of the public API. They remain reachable as
  `cfdmod.pressure.functions.*` / `cfdmod.pressure.geometry.*` for
  users who need them in tests, but only `run_*`, `apply_filters`,
  `calculate_statistics_from_h5`, the filter / config types, and the
  zoning / body / statistics models are exposed from
  `cfdmod.pressure` and the top-level package. Same goes for the
  `cfdmod.process_Cf` / `process_Cm` / `process_Ce` top-level
  re-exports that used to exist; those are gone.
- The `cfdmod.analysis` package was removed. Inflow lives in a
  single top-level module now, `cfdmod.inflow`. Migration:
    - `from cfdmod.analysis.inflow.profile import InflowData,
      NormalizationParameters` -> `from cfdmod.inflow import
      InflowData, NormalizationParameters`
    - `from cfdmod.analysis.inflow.functions import
      calculate_mean_velocity, ...` -> `from cfdmod.inflow import
      calculate_mean_velocity, ...`
  The top-level `from cfdmod import InflowData,
  NormalizationParameters` re-export is unchanged.

### Compatibility / migration

- Legacy pandas-HDFStore inputs from inflow (`cfdmod.inflow`) and HFPI
  (`cfdmod.hfpi.static.read_static_forces`) are read with a
  `DeprecationWarning`; the readers prefer the new layout but accept
  the old one.
- `cfdmod.pressure.migrate.migrate_body_h5` and `migrate_probe_h5` convert
  legacy pandas-HDFStore body/probe files to the new XDMF+H5 layout
  on disk for users who want to upgrade their fixtures.
- `aerosim-lnas` upgraded to `>=0.6.9,<0.7`.

### Documentation / tooling

- Top-level `notebooks/process_container_pack.ipynb` is the worked
  example: reads `bodies.body_cp body.h5` + `points.point_cp ref.h5`
  from the repo root, auto-detects container partition via a >1 m gap
  rule, runs Cp/Cf/Cm end-to-end with `lever_strategy="region_base"`
  (footprint-base lever) by default, and never authors a surface label
  (geometry is read straight from the body H5). Cf and Cm runs in the
  notebook are configured for `x` and `y` only -- the most common
  client-facing case.
- `cfdmod.notebook_utils` provides `mesh_summary`, `show_config`,
  `load_lnas` for exploratory notebook work.
- ASCII-only convention is now project-wide (CLAUDE.md):
  no em-dash, ellipsis, arrow glyphs, typographic quotes, or other
  non-ASCII characters in source, configs, notebooks or docs. Use
  `--`, `...`, `->`, `'`/`"` instead. The codebase was swept to
  match.

### Tests / quality

- pytest markers for the suite: `unit` (pure-function, fast),
  `integration` (multi-component end-to-end), and `perf` (synthetic
  big-data benchmarks). Default invocation excludes `perf`; run it
  explicitly with `pytest -m perf`.
- Shared pressure conftest centralises fixture paths, config
  builders (`make_cp_cfg`, `make_cf_cfg`, `make_cm_cfg`), zoning
  helpers and stats walkers, replacing the per-test boilerplate.
- Performance harness (`tests/pressure/test_perf.py`) synthesises
  body + probe data in-fixture (no dependency on root files) and
  drives the full Cp / Cf / Cm chain at two scales: medium
  (~30k triangles x 2k timesteps) and extreme (~150k triangles x
  10k timesteps, 1/5x of the worst real-world case).
- Per-run perf report: tracemalloc-tracked Python heap peak +
  `getrusage` RSS for each scale, written to
  `output/perf/perf_report.md` and `perf_report.json` for tracking
  regressions across releases.

### Dependencies / packaging

- Runtime deps slimmed: `tables` and `filelock` removed from the
  base install. `tables` is still needed for the legacy
  pandas-HDFStore compat readers in `cfdmod.inflow`,
  `cfdmod.hfpi.static`, and `cfdmod.pressure.migrate`; install via
  the new `legacy` extras (`pip install aerosim-cfdmod[legacy]`).
- `geometry` extras now ships only `trimesh`. `pymeshlab` was
  removed entirely -- it is GPL-licensed and would force GPL on any
  downstream code linking it. Code paths that genuinely need
  pymeshlab are documented and the user installs it explicitly at
  their own license risk.
- `docs` extras switched from `sphinx-book-theme` to `shibuya`
  (modern theme, active upstream).
- `cfdmod.io` now lazy-loads its vtk-backed helpers via PEP 562
  `__getattr__`. `import cfdmod` no longer pulls VTK at load time;
  accessing a vtk-backed name without the `vtk` extras raises a
  clear `ImportError` pointing at `pip install
  aerosim-cfdmod[vtk]`.
- `cfdmod.io.vtk.*` imports VTK classes via `vtkmodules.*`
  submodules (e.g. `from vtkmodules.vtkIOXML import
  vtkXMLPolyDataWriter`) instead of the catch-all `import vtk`,
  avoiding an unnecessary load of the full VTK universe on first
  use.
- Runtime dep floors bumped to the modern stack: `numpy>=2.0`,
  `scipy>=1.13`, `pandas>=2.2`, `pydantic>=2.10`, `matplotlib>=3.9`.
  Out-of-cap upper bounds relaxed to allow the current latest:
  `ruamel-yaml<0.20`, `pyarrow<25`, `myst-parser<6`, `ipython<10`,
  `ipykernel<8`, `pyvista<0.50`. Dev tooling brought to current
  majors: `pytest>=9`, `black>=26`, `isort>=8`, `ruff>=0.15`,
  `tox>=4.53`. `requires-python` stays at `>=3.10`; bump it (and
  `tables`) when 3.10 support is dropped.

## v1.1.2

- Automated CI/CD workflow

## v1.1.1

Coefficient time series are now in normalized time scales.
Time values from the solver are normalized by the CST value in the solver time scale.

- Input body pressure data is normalized by the CST value
- Derived coefficients also use a normalized time scale
- Parameters for statistical model are in full scale and need to be normalized using full scale CST.

## v1.1.0

It features the refactor for pressure use case module.

- Updated time series format to matrix form
- Changed how direction logic is applied for Cf and Cm
- Updated statistics functions and models

## v1.0.0

First production stable release.
It features all consulting use cases:
