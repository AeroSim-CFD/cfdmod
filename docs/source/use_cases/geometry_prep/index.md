# Triangle Grouping and Regrouping

Before pressures can be reduced to force, moment or shape coefficients, the
mesh triangles have to be **partitioned into meaningful groups** -- one group
per body, per facade, per floor, per zoning cell. cfdmod treats grouping as a
first-class, composable step, and provides two disk-level companions that act
on the result: **regroup** (rewrite a mesh + its timeseries around a new
grouping) and **remesh** (coarsen a grouped mesh).

The Python surface is documented in the
{doc}`API reference </api_reference>` (grouping, regroup and remesh sections);
this page explains **when to reach for which**.

## The grouping model

A grouping pipeline partitions or selects the triangles of a parent
`LnasFormat` mesh into named groups. Specs are Pydantic models in a
discriminated union ({data}`~cfdmod.GroupingSpec`), composed left-to-right
with {func}`~cfdmod.apply_groupings`, and a triangle may belong to **zero, one,
or many** groups. Each spec carries a `restrict_to` field so a later step can
scope its work to triangles already placed by an earlier one -- this is how a
`surface -> sub-body` nesting is reproduced.

The built-in kinds fall into a few intents:

- **By named geometry** -- {class}`~cfdmod.BySurfaceGrouping` collects named
  LNAS surfaces into groups.
- **By a regular grid** -- {class}`~cfdmod.ByZoningGrouping` bins centroids
  into a Cartesian grid; {class}`~cfdmod.ByDivisionsGrouping` (n cells per axis)
  and {class}`~cfdmod.BySizeGrouping` (fixed cell size) are convenience
  wrappers over it.
- **By orientation / geometry** -- {class}`~cfdmod.ByNormalGrouping`
  (windward / leeward / roof / sidewall by outward normal),
  {class}`~cfdmod.ByPlaneGrouping` (signed distance from an oriented plane),
  {class}`~cfdmod.ByCylindricalGrouping` (r, theta, axial bins -- towers,
  silos, chimneys).
- **By connectivity** -- {class}`~cfdmod.ByConnectivityGrouping` puts each
  shared-edge connected component in its own group (one physical body per
  component, no axis projection).
- **By statistics** -- {class}`~cfdmod.ByPercentileGrouping` makes equal-count
  quantile bins along an axis, for stable per-cell statistics when triangle
  density is uneven.
- **Escape hatch** -- {class}`~cfdmod.CustomGrouping` runs a user callback when
  none of the built-ins express the partition.

The force / moment / shape recipes are built on this abstraction: the
`body_grouping` / `zoning_grouping` / `connectivity_grouping` ops attach
a grouping to a {class}`~cfdmod.SurfaceDataSource`, and per-group series are
aggregated onto a {class}`~cfdmod.GroupsDataSource`.

## Regroup: a new mesh + aligned series

{func}`~cfdmod.run_regroup` (module `cfdmod.regroup`, runnable as
`python -m cfdmod.regroup`) takes a geometry plus a per-triangle HDF5
timeseries (Cp-style: rows are timesteps, columns are parent triangle ids),
applies a chain of grouping specs, and writes **two aligned outputs**:

- a new `LnasFormat` mesh with one named surface per group, and
- a new HDF5 timeseries whose columns line up with the new triangle order
  (`per_triangle`) or whose values are area-weighted per-group aggregates
  broadcast over each group's triangles (`area_weighted_mean`).

Reach for regroup when a downstream tool needs the grouping **baked into the
mesh on disk** (one surface per region) rather than applied on the fly -- for
example to hand a regrouped, per-facade mesh to ParaView or to a separate
coefficient run. It adds one regroup-local spec,
{class}`~cfdmod.BySizeRoundedPerComponent`, that fans out per-component
target-size subdivisions.

## Remesh: coarsen a grouped mesh

{func}`~cfdmod.remesh_per_group` (module `cfdmod.remesh`) coarsens a grouped
`LnasFormat` -- typically regroup output, where a group holds many triangles
from sliced cuts. Two paths:

- **Exact coplanar-fan collapse** (default) -- within each group, adjacent
  coplanar triangles are replaced by the minimum triangulation of their joined
  region. Lossless, deterministic, numpy-only: a flat NxN-subdivided square
  collapses to 2 triangles, a curved patch is left unchanged.
- **QEM decimation** ({func}`~cfdmod.decimate_qem`) -- opt-in Quadric Error
  Metrics decimation via the `remesh` extra, for curved groups where
  coplanar merge cannot reduce the count further.

Use remesh to shrink a mesh for faster rendering or downstream I/O once the
grouping is fixed; keep the default coplanar path unless a group is curved.

:::{seealso}
The grouping, regroup and remesh API sections in
{doc}`/api_reference`, and the `Cf` / `Cm` / `Ce` recipes in
{doc}`/use_cases/pressure/index` that consume groupings directly.
:::
