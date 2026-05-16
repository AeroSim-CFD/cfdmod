"""Remesh: per-group triangle coarsening for grouped LnasFormat meshes.

The default path is exact coplanar-fan merging: within each group, adjacent
triangles that share a plane are replaced by the minimum triangulation of
their joined region (a flat NxN-subdivided square becomes 2 triangles).
Lossless, deterministic, no external dependency.

An opt-in path runs Quadric Error Metrics (QEM) decimation on top via
``fast-simplification`` (MIT). Use it when groups lie on curved surfaces and
coplanar merge cannot reduce the triangle count further.

API surface only -- no CLI, no YAML config, no HDF5 I/O. Intended for
in-process use from notebooks and debugging scripts.
"""

__all__ = [
    "merge_coplanar",
    "decimate_qem",
    "remesh_per_group",
]

from cfdmod.remesh.functions import (
    decimate_qem,
    merge_coplanar,
    remesh_per_group,
)
