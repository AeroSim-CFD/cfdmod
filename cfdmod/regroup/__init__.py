"""Regroup: split / rearrange triangles + reorder a per-triangle timeseries.

A standalone preprocessing module that takes a geometry plus a per-triangle
HDF5 timeseries (Cp-style: rows = timesteps, columns = parent triangle ids),
applies a chain of triangle-grouping specs (Ce-style 90-degree zoning cuts,
connectivity-based container separation, etc.), and writes:

- a new ``LnasFormat`` mesh with one named surface per group, and
- a new HDF5 timeseries whose columns line up with the new triangle order
  (per_triangle mode) or whose values are area-weighted aggregates per group
  broadcast over each group's triangles (area_weighted_mean mode).

Reuses :mod:`cfdmod.geometry.grouping` for all binning kinds. Adds one
regroup-local spec, :class:`BySizeRoundedPerComponent`, that fans out
per-component target-size subdivisions (resolved by ``run_regroup``).
"""

__all__ = [
    "RegroupConfig",
    "BySizeRoundedPerComponent",
    "RegroupSpec",
    "RegroupIndex",
    "build_regroup_mapping",
    "build_regrouped_mesh",
    "apply_regroup_to_timeseries",
    "expand_regroup_chain",
    "run_regroup",
]

from cfdmod.regroup.functions import (
    RegroupIndex,
    apply_regroup_to_timeseries,
    build_regrouped_mesh,
    build_regroup_mapping,
)
from cfdmod.regroup.parameters import (
    BySizeRoundedPerComponent,
    RegroupConfig,
    RegroupSpec,
)
from cfdmod.regroup.run import expand_regroup_chain, run_regroup
