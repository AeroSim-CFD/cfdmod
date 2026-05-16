"""End-to-end container-pack post-processing pipeline.

:func:`run_container_pipeline` is the public entry point that mirrors
``notebooks/regroup_containers.ipynb``'s in-memory recipe:

1. Slice the body mesh along axis-aligned container-cell boundaries
   (``cfdmod.regroup`` ``aggregation="sliced"``).
2. Bucket fragments by ``(container_cell, axis_aligned_face_direction)``.
3. Coarsen each per-face surface via :func:`cfdmod.remesh.remesh_per_group`
   defaults (exact coplanar merge).
4. Stream Cp from ``body_h5`` + ``probe_h5`` through the same buckets,
   broadcasting the per-(container, face) area-weighted mean over the
   coarse triangles.

The whole geometric pipeline runs in memory; only the final coarse
``LnasFormat`` and per-face Cp timeseries land on disk. Notebooks and
tests both call this function so the implementation lives in one place.
"""

from __future__ import annotations

import pathlib
import time
from dataclasses import dataclass

import h5py
import numpy as np
from lnas import LnasFormat

from cfdmod.geometry.grouping import GroupingResult
from cfdmod.geometry.grouping.specs import ByConnectivityGrouping
from cfdmod.io.mesh import mesh_from_h5
from cfdmod.io.xdmf import filter_keys_by_range, get_pressure_keys, write_temporal_xdmf
from cfdmod.pressure.parameters import CpConfig
from cfdmod.regroup.functions import build_sliced_regrouped_mesh
from cfdmod.regroup.parameters import BySizeRoundedPerComponent, RegroupConfig
from cfdmod.regroup.run import build_regroup_mapping, expand_regroup_chain
from cfdmod.remesh.functions import remesh_per_group

__all__ = [
    "PipelineResult",
    "run_container_pipeline",
]


@dataclass
class PipelineResult:
    """Bundle of artefacts and counters returned by :func:`run_container_pipeline`."""

    remeshed_lnas: LnasFormat
    remeshed_h5: pathlib.Path
    remeshed_xdmf: pathlib.Path
    n_parents: int
    n_fragments: int
    n_coarse_triangles: int
    n_per_face_buckets: int
    n_timesteps: int
    timings: dict[str, float]


def run_container_pipeline(
    body_h5: pathlib.Path,
    probe_h5: pathlib.Path,
    output_dir: pathlib.Path,
    *,
    cp_config: CpConfig,
    target_size_x: float,
    target_size_y: float,
    target_size_z: float,
    min_triangles: int = 4,
) -> PipelineResult:
    """Run the in-memory container pipeline end-to-end.

    Args:
        body_h5: Body pressure XDMF+H5 (``/pressure/t{T}`` per timestep,
            ``/Triangles + /Geometry`` for the parent mesh).
        probe_h5: Reference probe XDMF+H5 (per-timestep reference pressure).
        output_dir: Destination directory; created if missing. Receives the
            three artefact files.
        cp_config: ``CpConfig`` carrying ``simul_U_H``, ``fluid_density``,
            ``macroscopic_type``, ``reference_pressure`` and (optionally)
            ``timestep_range``. The same model used by ``run_cp``.
        target_size_x, target_size_y, target_size_z: Per-axis container cell
            target sizes (metres) for the ``BySizeRoundedPerComponent``
            subdivision step. Each connected component is split into
            ``max(1, round(extent / target))`` cells along each axis.
        min_triangles: ``ByConnectivityGrouping`` threshold for dropping
            stray-triangle clusters.

    Returns:
        :class:`PipelineResult` with the coarse ``LnasFormat`` in memory,
        paths to the on-disk artefacts, intermediate cardinalities, and
        per-stage wall times.

    Outputs on disk (only these files; no full-cardinality intermediates):

    - ``{output_dir}/geometry.per_face.remeshed.lnas`` -- coarse mesh, one
      named surface per ``(container, axis_aligned_face)`` bucket.
    - ``{output_dir}/cp.per_face.remeshed.h5`` -- per-face Cp animation on
      the coarse mesh.
    - ``{output_dir}/cp.per_face.remeshed.xdmf`` -- sibling XDMF for
      ParaView.
    """
    output_dir = pathlib.Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    remeshed_lnas_path = output_dir / "geometry.per_face.remeshed.lnas"
    remeshed_h5 = output_dir / "cp.per_face.remeshed.h5"
    remeshed_xdmf = output_dir / "cp.per_face.remeshed.xdmf"
    if remeshed_h5.exists():
        remeshed_h5.unlink()

    timings: dict[str, float] = {}

    t0 = time.perf_counter()
    mesh = mesh_from_h5(body_h5)
    n_parents = mesh.geometry.triangles.shape[0]
    timings["load_mesh"] = time.perf_counter() - t0

    regroup_cfg = RegroupConfig(
        groupings=[
            ByConnectivityGrouping(name_template="cc{idx}", min_triangles=min_triangles),
            BySizeRoundedPerComponent(
                target_size_x=target_size_x,
                target_size_y=target_size_y,
                target_size_z=target_size_z,
                name_template="{parent}_c{idx}",
            ),
        ],
        aggregation="sliced",
        timeseries_group="cp",
        output_geometry_format="lnas",
        unassigned_policy="drop",
    )

    t0 = time.perf_counter()
    expanded, consumed, parent_intervals, parent_triangles = expand_regroup_chain(
        regroup_cfg.groupings, mesh, regroup_cfg.transformation
    )
    grouping = build_regroup_mapping(mesh, expanded, regroup_cfg.transformation)
    if consumed:
        grouping = GroupingResult(
            parent_n_triangles=grouping.parent_n_triangles,
            groups={n: i for n, i in grouping.groups.items() if n not in consumed},
        )
    sliced_lnas, regroup_index = build_sliced_regrouped_mesh(
        mesh,
        grouping,
        parent_intervals=parent_intervals,
        parent_triangles=parent_triangles,
        unassigned_policy="drop",
    )
    timings["slice"] = time.perf_counter() - t0
    n_fragments = sliced_lnas.geometry.triangles.shape[0]

    # Per-face bucketing (axis-aligned normal direction within each container).
    parent_of = regroup_index.new_to_parent
    tri_v = sliced_lnas.geometry.triangle_vertices
    crosses = np.cross(tri_v[:, 1] - tri_v[:, 0], tri_v[:, 2] - tri_v[:, 0])
    fragment_area = np.linalg.norm(crosses, axis=1) / 2
    norms = crosses / (np.linalg.norm(crosses, axis=1, keepdims=True) + 1e-30)
    axis_idx = np.abs(norms).argmax(axis=1)
    sign_neg = norms[np.arange(n_fragments), axis_idx] < 0
    direction = (axis_idx * 2 + sign_neg.astype(np.int64)).astype(np.int64)

    cell_names = sorted(sliced_lnas.surfaces.keys())
    cell_of = np.empty(n_fragments, dtype=np.int64)
    for ci, name in enumerate(cell_names):
        cell_of[sliced_lnas.surfaces[name]] = ci

    group_key = cell_of * 6 + direction
    unique_groups, bucket_of = np.unique(group_key, return_inverse=True)
    n_buckets = int(len(unique_groups))
    total_area_per_bucket = np.bincount(bucket_of, weights=fragment_area, minlength=n_buckets)
    safe_area = np.where(total_area_per_bucket > 0, total_area_per_bucket, 1.0)

    direction_suffix = {0: "xp", 1: "xn", 2: "yp", 3: "yn", 4: "zp", 5: "zn"}
    per_face_surfaces: dict[str, np.ndarray] = {}
    for gi, gkey in enumerate(unique_groups):
        per_face_surfaces[f"{cell_names[gkey // 6]}_{direction_suffix[gkey % 6]}"] = (
            np.flatnonzero(bucket_of == gi).astype(np.int32)
        )
    per_face_lnas = LnasFormat(
        version=sliced_lnas.version,
        geometry=sliced_lnas.geometry,
        surfaces=per_face_surfaces,
    )

    t0 = time.perf_counter()
    remeshed_lnas = remesh_per_group(per_face_lnas)
    timings["remesh"] = time.perf_counter() - t0
    n_out = remeshed_lnas.geometry.triangles.shape[0]

    output_tri_bucket = np.empty(n_out, dtype=np.int64)
    bucket_for_name = {
        f"{cell_names[g // 6]}_{direction_suffix[g % 6]}": gi for gi, g in enumerate(unique_groups)
    }
    for name, idxs in remeshed_lnas.surfaces.items():
        if idxs.size:
            output_tri_bucket[idxs] = bucket_for_name[name]

    remeshed_lnas.to_file(remeshed_lnas_path)

    # Cp transform constants drawn from cp_config so the public API stays in
    # sync with run_cp's formula.
    dynamic_pressure = 0.5 * cp_config.fluid_density * cp_config.simul_U_H**2
    multiplier = 1.0 / 3.0 if cp_config.macroscopic_type == "rho" else 1.0
    time_scale = (
        cp_config.simul_characteristic_length / cp_config.simul_U_H
        if cp_config.normalize_time
        else 1.0
    )

    keys = get_pressure_keys(body_h5, "pressure")
    if cp_config.timestep_range:
        keys = filter_keys_by_range(keys, cp_config.timestep_range)

    t0 = time.perf_counter()
    time_steps_arr: list[float] = []
    time_normalized_arr: list[float] = []
    with (
        h5py.File(body_h5, "r") as fb,
        h5py.File(probe_h5, "r") as fp,
        h5py.File(remeshed_h5, "w") as fd,
    ):
        fd.create_dataset("Triangles", data=remeshed_lnas.geometry.triangles.astype(np.int32))
        fd.create_dataset("Geometry", data=remeshed_lnas.geometry.vertices.astype(np.float64))
        cp_dst = fd.create_group("cp")
        for t_val, t_key in keys:
            p_body = fb["pressure"][t_key][:].astype(np.float64)
            probe_arr = fp["pressure"][t_key][:].astype(np.float64)
            if cp_config.reference_pressure == "probe":
                p_ref = float(probe_arr[0])
            elif cp_config.reference_pressure == "average":
                p_ref = float(probe_arr.mean())
            else:
                raise ValueError(f"unknown reference_pressure {cp_config.reference_pressure!r}")
            cp_parents = (p_body - p_ref) * (multiplier / dynamic_pressure)
            weighted = fragment_area * cp_parents[parent_of]
            bucket_sum = np.bincount(bucket_of, weights=weighted, minlength=n_buckets)
            bucket_cp = bucket_sum / safe_area
            cp_dst.create_dataset(t_key, data=bucket_cp[output_tri_bucket].astype(np.float64))
            time_steps_arr.append(t_val)
            time_normalized_arr.append(t_val / time_scale)
        meta = fd.create_group("meta")
        meta.create_dataset("time_steps", data=np.array(time_steps_arr))
        meta.create_dataset("time_normalized", data=np.array(time_normalized_arr))
    timings["stream_cp"] = time.perf_counter() - t0

    write_temporal_xdmf(remeshed_h5, remeshed_xdmf, "cp")
    timings["total"] = sum(v for k, v in timings.items() if k != "total")

    return PipelineResult(
        remeshed_lnas=remeshed_lnas,
        remeshed_h5=remeshed_h5,
        remeshed_xdmf=remeshed_xdmf,
        n_parents=n_parents,
        n_fragments=n_fragments,
        n_coarse_triangles=n_out,
        n_per_face_buckets=n_buckets,
        n_timesteps=len(keys),
        timings=timings,
    )
