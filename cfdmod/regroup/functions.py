"""Core algorithms for the regroup pipeline.

Three steps:

1. :func:`build_regroup_mapping` runs the (already-expanded) grouping chain
   against a transformed copy of the parent mesh, mirroring Ce's
   transformed-frame binning convention.
2. :func:`build_regrouped_mesh` emits a new :class:`LnasFormat` whose
   triangle order is the concatenation of group memberships (one named
   surface per group) and a :class:`RegroupIndex` describing how to
   build each output HDF5 column from the input.
3. :func:`apply_regroup_to_timeseries` streams the per-timestep HDF5
   transform: gather (per_triangle) or area-weighted mean broadcast
   (area_weighted_mean) over groups.

The output geometry contains exactly the parent triangles that fall in
at least one group (or all parent triangles if ``unassigned_policy=
"keep_as_unassigned"``). The output HDF5 has one column per output
triangle, so the geometry/timeseries cardinalities match for ParaView.
"""

from __future__ import annotations

import pathlib
from dataclasses import dataclass
from typing import Literal

import h5py
import numpy as np
from lnas import LnasFormat, LnasGeometry

from cfdmod.geometry.grouping import (
    GroupingResult,
    GroupingSpec,
    apply_groupings,
)
from cfdmod.io.geometry.transformation_config import TransformationConfig
from cfdmod.io.xdmf import (
    get_pressure_keys,
    read_step,
    read_timeseries_meta,
    write_timeseries_geometry,
    write_timeseries_meta,
    write_timeseries_step,
)
from cfdmod.logger import logger

__all__ = [
    "RegroupIndex",
    "build_regroup_mapping",
    "build_regrouped_mesh",
    "apply_regroup_to_timeseries",
]


_UNASSIGNED_NAME = "unassigned"


@dataclass(frozen=True)
class RegroupIndex:
    """Mapping from input HDF5 columns to output HDF5 columns.

    ``new_to_parent`` has length ``n_output_triangles`` and maps each
    output triangle position to the parent triangle index that fills it.

    ``aggregation`` controls how each output column is computed at write
    time:

    - ``"per_triangle"``: ``out[i, t] = in[new_to_parent[i], t]``.
    - ``"area_weighted_mean"``: every output triangle in group ``g`` gets
      the same value, ``sum(weights[g] * in[parents[g], t])``. Per-group
      ``parents`` and ``weights`` are stored in ``group_parents`` /
      ``group_weights``; ``triangle_group_idx[i]`` says which group entry
      output triangle ``i`` belongs to.
    """

    aggregation: str
    new_to_parent: np.ndarray
    output_group_names: list[str]
    triangle_group_idx: np.ndarray
    group_parents: list[np.ndarray]
    group_weights: list[np.ndarray]


def build_regroup_mapping(
    mesh: LnasFormat,
    groupings: list[GroupingSpec],
    transformation: TransformationConfig | None,
) -> GroupingResult:
    """Apply ``groupings`` against a transformed copy of ``mesh``.

    Mirrors Ce's transformed-frame binning convention: vertices are
    moved on a *copy* before binning, so the parent mesh's surfaces and
    triangle indices remain valid for the caller.
    """
    if transformation is None:
        return apply_groupings(mesh, groupings)
    mesh_for_binning = mesh.copy()
    mesh_for_binning.geometry.apply_transformation(
        transformation.get_geometry_transformation()
    )
    return apply_groupings(mesh_for_binning, groupings)


def _check_no_overlap(grouping: GroupingResult) -> None:
    """Raise if any parent triangle is assigned to more than one group."""
    membership = np.zeros(grouping.parent_n_triangles, dtype=np.int64)
    for idxs in grouping.groups.values():
        membership[np.asarray(idxs, dtype=np.int64)] += 1
    if membership.max() > 1:
        overlapping = int((membership > 1).sum())
        raise ValueError(
            f"regroup: per_triangle aggregation requires groups to partition "
            f"the parent mesh (no overlaps); {overlapping} triangle(s) appear "
            "in multiple groups. Use area_weighted_mean or fix the chain."
        )


def _augment_with_unassigned(
    grouping: GroupingResult,
) -> tuple[list[str], list[np.ndarray]]:
    """Return (group_names, group_parents) including an 'unassigned' bucket."""
    in_any = np.zeros(grouping.parent_n_triangles, dtype=bool)
    for idxs in grouping.groups.values():
        in_any[np.asarray(idxs, dtype=np.int64)] = True
    unassigned = np.flatnonzero(~in_any).astype(np.int64)
    names = list(grouping.groups.keys())
    parents = [np.asarray(grouping.groups[n], dtype=np.int64) for n in names]
    if unassigned.size:
        if _UNASSIGNED_NAME in grouping.groups:
            raise ValueError(
                f"regroup: cannot keep unassigned triangles, group name "
                f"{_UNASSIGNED_NAME!r} already exists in the chain output."
            )
        names.append(_UNASSIGNED_NAME)
        parents.append(unassigned)
    return names, parents


def build_regrouped_mesh(
    mesh: LnasFormat,
    grouping: GroupingResult,
    *,
    aggregation: Literal["per_triangle", "area_weighted_mean"],
    unassigned_policy: Literal["drop", "keep_as_unassigned"],
) -> tuple[LnasFormat, RegroupIndex]:
    """Build the output LnasFormat and the input->output column mapping.

    Output triangles appear in the concatenation order
    ``[group0_parents, group1_parents, ...]`` (insertion order of
    ``grouping.groups``, with an optional trailing ``unassigned`` bucket).
    Surfaces on the returned LnasFormat carry one entry per group.
    """
    if aggregation == "per_triangle":
        _check_no_overlap(grouping)

    if unassigned_policy == "keep_as_unassigned":
        names, parents = _augment_with_unassigned(grouping)
    else:
        names = list(grouping.groups.keys())
        parents = [np.asarray(grouping.groups[n], dtype=np.int64) for n in names]

    # Drop empty groups (apply_groupings already does, but augment may add empty if all assigned).
    keep = [(n, p) for n, p in zip(names, parents) if p.size > 0]
    if not keep:
        raise ValueError("regroup: chain produced zero non-empty groups.")
    names = [n for n, _ in keep]
    parents = [p for _, p in keep]

    # Build the new triangle order and per-output-triangle group index.
    new_to_parent = np.concatenate(parents).astype(np.int64)
    triangle_group_idx = np.concatenate(
        [np.full(p.size, gi, dtype=np.int64) for gi, p in enumerate(parents)]
    )

    parent_triangles = mesh.geometry.triangles  # (n_tri, 3) vertex indices
    parent_vertices = mesh.geometry.vertices

    new_triangles = parent_triangles[new_to_parent].copy()
    new_geom = LnasGeometry(
        vertices=parent_vertices.copy(),
        triangles=new_triangles,
    )

    # Build the new surfaces: each group is a contiguous block in new_to_parent.
    new_surfaces: dict[str, np.ndarray] = {}
    cursor = 0
    for name, parent_idxs in zip(names, parents):
        new_surfaces[name] = np.arange(
            cursor, cursor + parent_idxs.size, dtype=np.int32
        )
        cursor += parent_idxs.size

    new_lnas = LnasFormat(
        version=mesh.version,
        geometry=new_geom,
        surfaces=new_surfaces,
    )

    # Per-group area weights for area_weighted_mean (computed on parent geometry).
    if aggregation == "area_weighted_mean":
        parent_areas = mesh.geometry.areas
        group_weights: list[np.ndarray] = []
        for parent_idxs in parents:
            a = parent_areas[parent_idxs].astype(np.float64)
            total = a.sum()
            if total <= 0:
                # Degenerate group (e.g., all zero-area triangles); fall back
                # to uniform weights so the aggregate is well-defined.
                w = np.full(a.size, 1.0 / a.size, dtype=np.float64)
            else:
                w = a / total
            group_weights.append(w)
    else:
        group_weights = [np.empty(0, dtype=np.float64) for _ in parents]

    index = RegroupIndex(
        aggregation=aggregation,
        new_to_parent=new_to_parent,
        output_group_names=names,
        triangle_group_idx=triangle_group_idx,
        group_parents=parents,
        group_weights=group_weights,
    )
    return new_lnas, index


def _per_triangle_region_labels(index: RegroupIndex) -> list[str]:
    """One label per output triangle, naming the group it belongs to."""
    return [
        index.output_group_names[int(g)] for g in index.triangle_group_idx.tolist()
    ]


def apply_regroup_to_timeseries(
    input_h5: pathlib.Path,
    output_h5: pathlib.Path,
    *,
    group: str,
    regroup_index: RegroupIndex,
    new_triangles: np.ndarray,
    new_vertices: np.ndarray,
) -> None:
    """Stream-rewrite ``input_h5[group]`` to ``output_h5[group]`` per the index.

    Writes ``/Triangles + /Geometry`` (from the regrouped mesh), one
    ``/{group}/t{T}`` dataset per input timestep, and ``/meta`` carrying
    the original ``time_steps`` / ``time_normalized`` plus per-output-
    triangle ``region_labels`` (the group each triangle belongs to).
    """
    output_h5 = pathlib.Path(output_h5)
    if output_h5.exists():
        output_h5.unlink()
    output_h5.parent.mkdir(parents=True, exist_ok=True)

    write_timeseries_geometry(output_h5, new_triangles, new_vertices)

    in_meta = read_timeseries_meta(input_h5)
    region_labels = _per_triangle_region_labels(regroup_index)

    keys = get_pressure_keys(input_h5, group=group)
    if not keys:
        raise ValueError(
            f"regroup: input H5 {input_h5} has no timesteps under /{group}/"
        )

    n_parent_expected = int(regroup_index.new_to_parent.max()) + 1
    with h5py.File(input_h5, "r") as f:
        first_step = f[group][keys[0][1]][:]
    if first_step.shape[0] < n_parent_expected:
        raise ValueError(
            f"regroup: input H5 has {first_step.shape[0]} columns at "
            f"/{group}/{keys[0][1]} but the regroup index references parent "
            f"triangle index {n_parent_expected - 1}."
        )

    n_out = int(regroup_index.new_to_parent.size)
    logger.info(
        f"regroup: writing {len(keys)} timestep(s), {n_out} output triangle(s), "
        f"{len(regroup_index.output_group_names)} group(s) to {output_h5}"
    )

    if regroup_index.aggregation == "per_triangle":
        gather = regroup_index.new_to_parent
        for _t_val, key in keys:
            in_col = read_step(input_h5, key, group)
            out_col = in_col[gather]
            write_timeseries_step(output_h5, group, key, out_col)
    elif regroup_index.aggregation == "area_weighted_mean":
        triangle_group_idx = regroup_index.triangle_group_idx
        for _t_val, key in keys:
            in_col = read_step(input_h5, key, group)
            per_group = np.empty(
                len(regroup_index.output_group_names), dtype=np.float64
            )
            for gi, (parents, weights) in enumerate(
                zip(regroup_index.group_parents, regroup_index.group_weights)
            ):
                per_group[gi] = float(np.sum(weights * in_col[parents]))
            out_col = per_group[triangle_group_idx]
            write_timeseries_step(output_h5, group, key, out_col)
    else:
        raise ValueError(
            f"regroup: unknown aggregation {regroup_index.aggregation!r}"
        )

    write_timeseries_meta(
        output_h5,
        time_steps=in_meta["time_steps"],
        time_normalized=in_meta["time_normalized"],
        region_labels=region_labels,
    )
