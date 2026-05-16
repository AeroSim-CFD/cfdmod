"""Shared fixtures + a self-contained reference pipeline for ``tests/remesh``.

``run_container_pipeline`` mirrors the in-memory pipeline used by
``notebooks/regroup_containers.ipynb`` (slice + per-face buckets + remesh +
inline Cp stream). It is centralised here so both the integration test and
the perf test can drive it from the same code path.
"""

from __future__ import annotations

import pathlib
import time
from dataclasses import dataclass

import h5py
import numpy as np
from lnas import LnasFormat

from cfdmod import (
    ByConnectivityGrouping,
    BySizeRoundedPerComponent,
    RegroupConfig,
    mesh_from_h5,
)
from cfdmod.geometry.grouping import GroupingResult
from cfdmod.io.xdmf import get_pressure_keys, write_temporal_xdmf
from cfdmod.regroup import build_regroup_mapping, expand_regroup_chain
from cfdmod.regroup.functions import build_sliced_regrouped_mesh
from cfdmod.remesh import remesh_per_group

FIXTURE_DIR = pathlib.Path(__file__).resolve().parents[2] / "fixtures" / "tests" / "remesh"
FIXTURE_BODY = FIXTURE_DIR / "bodies.h5"
FIXTURE_PROBE = FIXTURE_DIR / "points.h5"
FIXTURE_MANIFEST = FIXTURE_DIR / "manifest.yaml"


@dataclass
class PipelineResult:
    """All artifacts produced by :func:`run_container_pipeline`."""

    remeshed_lnas: LnasFormat
    remeshed_h5: pathlib.Path
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
    target_size_x: float = 6.34,
    target_size_y: float = 2.58,
    target_size_z: float = 2.6,
    min_triangles: int = 4,
    simul_U_H: float = 1.0,
    fluid_density: float = 1.0,
    macroscopic_type: str = "pressure",
    reference_pressure: str = "probe",
) -> PipelineResult:
    """Run the in-memory container pipeline on a body+probe pair.

    Returns the coarse ``LnasFormat`` plus the path to the per-face Cp h5,
    a summary of intermediate cardinalities, and per-stage wall times.
    """
    output_dir = pathlib.Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
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

    # Per-face buckets.
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

    dynamic_pressure = 0.5 * fluid_density * simul_U_H**2
    multiplier = 1.0 / 3.0 if macroscopic_type == "rho" else 1.0

    keys = get_pressure_keys(body_h5, "pressure")

    t0 = time.perf_counter()
    time_steps = []
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
            if reference_pressure == "probe":
                p_ref = float(probe_arr[0])
            elif reference_pressure == "average":
                p_ref = float(probe_arr.mean())
            else:
                raise ValueError(f"unknown reference_pressure {reference_pressure!r}")
            cp_parents = (p_body - p_ref) * (multiplier / dynamic_pressure)
            weighted = fragment_area * cp_parents[parent_of]
            bucket_sum = np.bincount(bucket_of, weights=weighted, minlength=n_buckets)
            bucket_cp = bucket_sum / safe_area
            cp_dst.create_dataset(t_key, data=bucket_cp[output_tri_bucket].astype(np.float64))
            time_steps.append(t_val)
        meta = fd.create_group("meta")
        meta.create_dataset("time_steps", data=np.array(time_steps))
        meta.create_dataset("time_normalized", data=np.array(time_steps))
    timings["stream_cp"] = time.perf_counter() - t0

    write_temporal_xdmf(remeshed_h5, remeshed_xdmf, "cp")
    timings["total"] = sum(v for k, v in timings.items() if k != "total")

    return PipelineResult(
        remeshed_lnas=remeshed_lnas,
        remeshed_h5=remeshed_h5,
        n_parents=n_parents,
        n_fragments=n_fragments,
        n_coarse_triangles=n_out,
        n_per_face_buckets=n_buckets,
        n_timesteps=len(keys),
        timings=timings,
    )
