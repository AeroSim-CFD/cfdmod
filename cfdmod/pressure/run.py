"""Orchestration functions for the pressure module.

Pure Python entry points called by cli.py. No argparse or file-path logic here.

Pipeline contract: every coefficient first persists its full per-triangle
timeseries to disk (XDMF+H5), then computes statistics from that on-disk
file via cfdmod.pressure.statistics_runner.calculate_statistics_from_h5.
Stats are appended to a single combined stats.h5 with an embedded mesh
per leaf group (so different sub-meshes - body subsets for Cf/Cm, sliced
regions mesh for Ce - coexist without length collisions).
"""

from __future__ import annotations

__all__ = ["run_cp", "run_cf", "run_cm", "run_ce"]

import pathlib

import h5py
import numpy as np
import pandas as pd
from lnas import LnasFormat

from cfdmod.io.mesh import load_mesh, mesh_from_h5
from cfdmod.io.xdmf import (
    read_processing_metadata,
    read_timeseries_meta,
    write_processing_metadata,
    write_stats_field,
    write_stats_xdmf,
    write_temporal_xdmf,
    write_timeseries_geometry,
    write_timeseries_meta,
    write_timeseries_step,
)
from cfdmod.logger import logger
from cfdmod.pressure.functions import process_Ce, process_Cf, process_Cm, process_xdmf_to_cp
from cfdmod.pressure.parameters import (
    CeCaseConfig,
    CfCaseConfig,
    CmCaseConfig,
    CpCaseConfig,
    ExtremeGumbelParamsModel,
    ParameterizedStatisticModel,
)
from cfdmod.pressure.path_manager import CePathManager, CfPathManager, CmPathManager, CpPathManager
from cfdmod.pressure.statistics_runner import calculate_statistics_from_h5
from cfdmod.utils import create_folders_for_file


def _coerce_case_cfg(source, model_cls):
    """Resolve ``cfg_path`` to a case-config instance.

    Accepts either an already-built case-config (returned as-is, useful for
    in-process pipelines) or a path to a YAML file (loaded via
    ``model_cls.from_file``).
    """
    if isinstance(source, model_cls):
        return source
    return model_cls.from_file(pathlib.Path(source))


def _cp_time_steps_for(cp_h5: pathlib.Path, time_normalized: np.ndarray) -> np.ndarray:
    """Given a Cf/Cm/Ce-derived ``time_normalized`` array, return the matching
    ``time_steps`` (raw solver time) values from cp_h5's /meta.

    Cf/Cm/Ce do not write their own time_steps; they inherit the time axis
    Cp persisted. With ``CpConfig.normalize_time=False`` (the default) the
    two arrays are identical, but with ``normalize_time=True`` they are not,
    and writing the same array into both slots silently mislabels the
    output. This helper recovers the raw partner via /meta lookup.
    """
    meta = read_timeseries_meta(cp_h5)
    cp_norm = np.asarray(meta["time_normalized"], dtype=np.float64)
    cp_raw = np.asarray(meta["time_steps"], dtype=np.float64)
    if cp_norm.size == cp_raw.size and np.allclose(cp_norm, cp_raw):
        return np.asarray(time_normalized, dtype=np.float64)
    norm_to_raw = {float(n): float(r) for n, r in zip(cp_norm, cp_raw)}
    return np.array([norm_to_raw[float(t)] for t in time_normalized], dtype=np.float64)


def _read_cp_simul_scale(cp_h5: pathlib.Path) -> tuple[float | None, float | None]:
    """Return (simul_U_H, simul_characteristic_length) from cp_h5's metadata.

    Returns (None, None) if the file has no /processing_metadata or the
    Cp config didn't carry these fields (legacy inputs).
    """
    try:
        meta = read_processing_metadata(cp_h5, "/")
    except (KeyError, OSError):
        return None, None
    cfg = meta.get("config") or {}
    u_h = cfg.get("simul_U_H")
    char_len = cfg.get("simul_characteristic_length")
    return (
        float(u_h) if u_h is not None else None,
        float(char_len) if char_len is not None else None,
    )


def _inherit_full_scale_from_cp(case_cfg, cp_h5: pathlib.Path) -> None:
    """Fill in any unset ``full_scale_*`` on Gumbel / Moving-Average stats
    inside ``case_cfg`` by reading the simul-scale values persisted on cp_h5.

    Mutates the case config in place. Only touches fields the user left as
    ``None``; explicit user values are preserved. Statistics that don't
    need full-scale fields (mean, min, max, ...) are untouched.
    """
    cp_u_h: float | None = None
    cp_char_len: float | None = None
    cached = False

    coef_attr = next(
        (
            a
            for a in (
                "force_coefficient",
                "moment_coefficient",
                "shape_coefficient",
                "pressure_coefficient",
            )
            if hasattr(case_cfg, a)
        ),
        None,
    )
    if coef_attr is None:
        return
    coef_dict = getattr(case_cfg, coef_attr)

    for cfg in coef_dict.values():
        for stat in cfg.statistics:
            if not isinstance(stat, ParameterizedStatisticModel):
                continue
            params = stat.params
            if not isinstance(params, ExtremeGumbelParamsModel):
                continue
            needs_u = params.full_scale_U_H is None
            needs_l = params.full_scale_characteristic_length is None
            if not (needs_u or needs_l):
                continue
            if not cached:
                cp_u_h, cp_char_len = _read_cp_simul_scale(cp_h5)
                cached = True
            if needs_u and cp_u_h is not None:
                params.full_scale_U_H = cp_u_h
            if needs_l and cp_char_len is not None:
                params.full_scale_characteristic_length = cp_char_len


def _write_region_timeseries(
    ts_path: pathlib.Path,
    *,
    triangles: np.ndarray,
    vertices: np.ndarray,
    region_idx_per_tri: np.ndarray,
    data_df: pd.DataFrame,
    group: str,
) -> np.ndarray:
    """Broadcast a per-region timeseries (rows=time, cols=region_idx) to
    per-triangle values aligned with the given mesh, and append each timestep
    as ``/{group}/t{T}`` to ``ts_path``. Returns the time array (= data_df.index)
    so callers can write meta and the temporal XDMF without re-deriving it.

    The mesh datasets (``/Triangles``, ``/Geometry``) are written once if not
    already present in the file.
    """
    with h5py.File(ts_path, "a") as f:
        has_geom = "Triangles" in f and "Geometry" in f
    if not has_geom:
        write_timeseries_geometry(ts_path, triangles, vertices)

    # Build per-triangle column indices into data_df. For zoning splits where
    # the regions mesh is finer than the data (e.g. a slice plane lands inside
    # a region that has no Cp samples), the missing tri positions are filled
    # with NaN - same behaviour as process_surfaces' existing left-join warn.
    col_pos = {c: i for i, c in enumerate(data_df.columns)}
    tri_col_idx = np.array([col_pos.get(r, -1) for r in region_idx_per_tri], dtype=np.int64)
    missing_mask = tri_col_idx < 0

    data_arr = data_df.to_numpy(dtype=np.float64)
    safe_idx = np.where(missing_mask, 0, tri_col_idx)
    per_tri = data_arr[:, safe_idx]  # (n_time, n_tri)
    if missing_mask.any():
        per_tri[:, missing_mask] = np.nan

    times = data_df.index.to_numpy(dtype=np.float64)
    for i, t_norm in enumerate(times):
        write_timeseries_step(ts_path, group, f"t{t_norm}", per_tri[i])

    return times


def _write_stats_for_group(
    stats_h5: pathlib.Path,
    *,
    timeseries_path: pathlib.Path,
    timeseries_group: str,
    stats_group: str,
    statistics,
    triangles: np.ndarray,
    vertices: np.ndarray,
) -> None:
    """Compute stats from an on-disk timeseries group and append to stats.h5
    under ``stats_group`` with the embedded mesh."""
    stats_df = calculate_statistics_from_h5(
        h5_path=timeseries_path,
        group=timeseries_group,
        statistics=statistics,
        timestep_range=None,
    )
    for stat_name in stats_df.columns:
        write_stats_field(
            h5_path=stats_h5,
            group=stats_group,
            stat_name=stat_name,
            values=stats_df[stat_name].to_numpy(dtype=np.float64),
            triangles=triangles,
            vertices=vertices,
        )


def run_cp(
    body_h5: pathlib.Path,
    probe_h5: pathlib.Path | None,
    cfg_path: pathlib.Path | str | CpCaseConfig | CfCaseConfig | CmCaseConfig | CeCaseConfig,
    output: pathlib.Path,
    mesh_path: pathlib.Path | LnasFormat | None = None,
) -> None:
    """Compute Cp timeseries + stats.

    Outputs per config label:

    - ``{output}/cp.{label}.time_series.h5`` + ``.xdmf`` -- timeseries
    - ``{output}/stats.h5`` + ``stats.xdmf`` -- stats; embeds the full
      mesh under ``/cp/{label}/`` alongside stat datasets

    Args:
        body_h5: Body pressure XDMF+H5.
        probe_h5: Reference probe XDMF+H5 (or None).
        cfg_path: Cp YAML config path.
        output: Output directory.
        mesh_path: Optional geometry source -- ``.lnas``, ``.stl``, ``.h5``,
            ``.xdmf``, or a pre-loaded :class:`LnasFormat`. If omitted, the
            geometry is read from ``body_h5`` itself (single ``"all"`` surface).
    """
    case_cfg = _coerce_case_cfg(cfg_path, CpCaseConfig)
    mesh = mesh_from_h5(body_h5) if mesh_path is None else load_mesh(mesh_path)
    path_manager = CpPathManager(output_path=output)
    stats_h5 = path_manager.get_stats_h5_path()
    create_folders_for_file(stats_h5)

    triangles = mesh.geometry.triangles
    vertices = mesh.geometry.vertices
    # When the user supplied an external reference mesh, embed it in the cp_h5
    # output instead of the body H5's per-direction coordinates. Downstream
    # (Cf/Cm/Ce) reads geometry from cp_h5 by default, so the reference frame
    # propagates without the user having to re-pass mesh_path each call.
    mesh_override = (triangles, vertices) if mesh_path is not None else None

    for cfg_lbl, cfg in case_cfg.pressure_coefficient.items():
        logger.info(f"Processing Cp: {cfg_lbl}")

        timeseries_path = path_manager.get_timeseries_path(cfg_lbl=cfg_lbl)
        create_folders_for_file(timeseries_path)

        logger.info("Transforming to Cp timeseries...")
        process_xdmf_to_cp(
            body_h5=body_h5,
            probe_h5=probe_h5,
            output_path=timeseries_path,
            cp_config=cfg,
            mesh_override=mesh_override,
        )

        cfg_dump = cfg.model_dump()
        ts_inputs = {
            "body_h5": str(body_h5),
            "probe_h5": str(probe_h5),
            "mesh_path": str(mesh_path) if mesh_path is not None else f"<from {body_h5}>",
        }
        write_processing_metadata(
            timeseries_path,
            "/",
            cfg_dump,
            extra={"coefficient": "cp", "cfg_lbl": cfg_lbl, **ts_inputs},
        )

        if cfg.statistics:
            logger.info("Calculating Cp statistics from on-disk timeseries...")
            _write_stats_for_group(
                stats_h5,
                timeseries_path=timeseries_path,
                timeseries_group="cp",
                stats_group=f"cp/{cfg_lbl}",
                statistics=cfg.statistics,
                triangles=triangles,
                vertices=vertices,
            )
            write_processing_metadata(
                stats_h5,
                f"cp/{cfg_lbl}",
                cfg_dump,
                extra={"coefficient": "cp", "cfg_lbl": cfg_lbl, **ts_inputs},
            )

            write_stats_xdmf(stats_h5, path_manager.get_stats_xdmf_path())
            logger.info(f"Cp stats written for config '{cfg_lbl}'")
        else:
            logger.info(f"Cp '{cfg_lbl}': statistics list is empty; skipping stats step")


def _run_body_coefficient(
    *,
    coef: str,
    mesh: LnasFormat,
    cp_h5: pathlib.Path,
    cfg_lbl: str,
    cfg,
    bodies_definition,
    process_fn,
    path_manager,
    stats_h5: pathlib.Path,
) -> None:
    """Shared body-coefficient flow for Cf and Cm.

    1. Run the in-memory transform (process_Cf or process_Cm) to get per-region
       Cf/Cm matrices keyed by direction.
    2. For each body: write a per-body timeseries H5 with one group per
       direction (cf_x/cf_y/cf_z), broadcasting the per-region values onto
       body-mesh triangles via the region-indexing df produced upstream.
    3. Compute stats from the on-disk file and append to stats.h5 under
       /{coef}_{dir}/{cfg_lbl}/{body}/ with the body mesh embedded.
    """
    compiled_output = process_fn(
        mesh=mesh,
        cfg=cfg,
        cp_h5=cp_h5,
        bodies_definition=bodies_definition,
    )

    geometry_df = compiled_output[cfg.directions[0]].region_indexing_df

    for body_cfg in cfg.bodies:
        sfc_list = bodies_definition[body_cfg.name].surfaces
        # Empty surfaces list means "every surface in the mesh" -- the same
        # convention get_geometry_data uses upstream. This makes the synthetic-
        # surface path (loaded from a body H5 / STL with one "all" surface)
        # work without forcing the user to know its name.
        if not sfc_list:
            sfc_list = list(mesh.surfaces.keys())
        body_geom = mesh.geometry_from_list_surfaces(surfaces_names=sfc_list)[0]
        body_region_idx = geometry_df.loc[
            geometry_df.region_idx.str.contains(body_cfg.name)
        ].region_idx.to_numpy()

        ts_path = path_manager.get_body_timeseries_path(cfg_lbl=cfg_lbl, body_name=body_cfg.name)
        create_folders_for_file(ts_path)
        if ts_path.exists():
            ts_path.unlink()

        write_timeseries_geometry(ts_path, body_geom.triangles, body_geom.vertices)

        times: np.ndarray | None = None
        for direction in cfg.directions:
            data_df = compiled_output[direction].data_df
            times = _write_region_timeseries(
                ts_path,
                triangles=body_geom.triangles,
                vertices=body_geom.vertices,
                region_idx_per_tri=body_region_idx,
                data_df=data_df,
                group=f"{coef}_{direction}",
            )

        assert times is not None
        write_timeseries_meta(ts_path, _cp_time_steps_for(cp_h5, times), times)
        write_temporal_xdmf(
            ts_path,
            ts_path.with_suffix(".xdmf"),
            group=[f"{coef}_{d}" for d in cfg.directions],
        )

        cfg_dump = cfg.model_dump()
        meta_extra = {
            "coefficient": coef,
            "cfg_lbl": cfg_lbl,
            "body": body_cfg.name,
            "cp_h5": str(cp_h5),
        }
        write_processing_metadata(ts_path, "/", cfg_dump, extra=meta_extra)

        if cfg.statistics:
            for direction in cfg.directions:
                stats_grp = f"{coef}_{direction}/{cfg_lbl}/{body_cfg.name}"
                _write_stats_for_group(
                    stats_h5,
                    timeseries_path=ts_path,
                    timeseries_group=f"{coef}_{direction}",
                    stats_group=stats_grp,
                    statistics=cfg.statistics,
                    triangles=body_geom.triangles,
                    vertices=body_geom.vertices,
                )
                write_processing_metadata(
                    stats_h5,
                    stats_grp,
                    cfg_dump,
                    extra={**meta_extra, "direction": direction},
                )


def run_cf(
    cp_h5: pathlib.Path,
    cfg_path: pathlib.Path | str | CpCaseConfig | CfCaseConfig | CmCaseConfig | CeCaseConfig,
    output: pathlib.Path,
    mesh_path: pathlib.Path | LnasFormat | None = None,
) -> None:
    """Compute Cf per direction + stats.

    Per body and cfg label:

    - ``{output}/Cf.{label}.{body}.time_series.h5`` + ``.xdmf`` -- one group
      per direction (``/cf_x``, ``/cf_y``, ``/cf_z``), each with ``/t{T}``
      arrays of length ``n_body_tri``.
    - ``stats.h5``: ``/cf_{dir}/{label}/{body}/{Triangles, Geometry,
      stat...}``.

    ``mesh_path`` is optional and accepts ``.lnas``, ``.stl``, ``.h5``,
    ``.xdmf``, or a pre-loaded :class:`LnasFormat`. If omitted, the geometry
    is read from ``cp_h5`` itself (single ``"all"`` surface).
    """
    case_cfg = _coerce_case_cfg(cfg_path, CfCaseConfig)
    _inherit_full_scale_from_cp(case_cfg, cp_h5)
    mesh = mesh_from_h5(cp_h5) if mesh_path is None else load_mesh(mesh_path)
    path_manager = CfPathManager(output_path=output)
    stats_h5 = path_manager.get_stats_h5_path()
    create_folders_for_file(stats_h5)

    for cfg_lbl, cfg in case_cfg.force_coefficient.items():
        logger.info(f"Processing Cf: {cfg_lbl}")
        _run_body_coefficient(
            coef="cf",
            mesh=mesh,
            cp_h5=cp_h5,
            cfg_lbl=cfg_lbl,
            cfg=cfg,
            bodies_definition=case_cfg.bodies,
            process_fn=process_Cf,
            path_manager=path_manager,
            stats_h5=stats_h5,
        )
        if cfg.statistics and stats_h5.exists():
            write_stats_xdmf(stats_h5, path_manager.get_stats_xdmf_path())
            logger.info(f"Cf stats written for config '{cfg_lbl}'")
        else:
            logger.info(f"Cf '{cfg_lbl}': statistics list is empty; skipping stats step")


def _bbox_corners_xy_cases(
    body_cfg, mesh: LnasFormat, sfc_list: list[str]
) -> dict[str, dict[int, tuple[float, float, float]]]:
    """For a body's regions, generate 4 lever-origin candidates per region at
    the xy corners of each region's bounding box (z = min z of the region).

    Returns a dict mapping case label (xmin_ymin, xmin_ymax, xmax_ymin,
    xmax_ymax) to a per-region-int origin map.
    """
    from cfdmod.pressure.geometry import get_indexing_mask

    body_geom = mesh.geometry_from_list_surfaces(surfaces_names=sfc_list)[0]
    transformed = body_geom.copy()
    df_regions = body_cfg.sub_bodies.get_regions_df()
    region_idx_per_tri = get_indexing_mask(mesh=transformed, df_regions=df_regions)

    cases: dict[str, dict[int, tuple[float, float, float]]] = {
        "xmin_ymin": {},
        "xmin_ymax": {},
        "xmax_ymin": {},
        "xmax_ymax": {},
    }
    for region_int in np.unique(region_idx_per_tri):
        if region_int < 0:
            continue
        in_region = np.where(region_idx_per_tri == region_int)[0]
        xyz = transformed.triangle_vertices[in_region]
        x_min = float(xyz[:, :, 0].min())
        x_max = float(xyz[:, :, 0].max())
        y_min = float(xyz[:, :, 1].min())
        y_max = float(xyz[:, :, 1].max())
        z_min = float(xyz[:, :, 2].min())
        cases["xmin_ymin"][int(region_int)] = (x_min, y_min, z_min)
        cases["xmin_ymax"][int(region_int)] = (x_min, y_max, z_min)
        cases["xmax_ymin"][int(region_int)] = (x_max, y_min, z_min)
        cases["xmax_ymax"][int(region_int)] = (x_max, y_max, z_min)
    return cases


def _expand_moment_cases(cfg, bodies_definition: dict, mesh: LnasFormat) -> list[tuple]:
    """Resolve a moment cfg's bodies into one independent run per
    (body, lever-origin case) pair.

    Each entry in the returned list is a ``(single_body_cfg, bodies_def)``
    tuple ready to feed into ``_run_body_coefficient`` -- always exactly one
    body per run so the downstream geometry/region machinery never sees
    triangles re-registered under multiple body names.

    Bodies with no cases (no ``lever_origin_cases`` and no
    ``region_bbox_corners_xy`` strategy) pass through unchanged as a single
    run with the original body name.
    """
    runs: list[tuple] = []

    for body_cfg in cfg.bodies:
        sfc_list = bodies_definition[body_cfg.name].surfaces or list(mesh.surfaces.keys())

        if body_cfg.lever_origin_cases:
            case_map = body_cfg.lever_origin_cases
        elif body_cfg.lever_strategy == "region_bbox_corners_xy":
            case_map = _bbox_corners_xy_cases(body_cfg, mesh, sfc_list)
        else:
            single_cfg = cfg.model_copy(update={"bodies": [body_cfg]})
            single_def = {body_cfg.name: bodies_definition[body_cfg.name]}
            runs.append((single_cfg, single_def))
            continue

        for case_label, region_origins in case_map.items():
            new_name = f"{body_cfg.name}.{case_label}" if case_label else body_cfg.name
            derived = body_cfg.model_copy(
                update={
                    "name": new_name,
                    "lever_strategy": "fixed",
                    "region_lever_origins": region_origins,
                    "lever_origin_cases": None,
                }
            )
            single_cfg = cfg.model_copy(update={"bodies": [derived]})
            single_def = {new_name: bodies_definition[body_cfg.name]}
            runs.append((single_cfg, single_def))

    return runs


def run_cm(
    cp_h5: pathlib.Path,
    cfg_path: pathlib.Path | str | CpCaseConfig | CfCaseConfig | CmCaseConfig | CeCaseConfig,
    output: pathlib.Path,
    mesh_path: pathlib.Path | LnasFormat | None = None,
) -> None:
    """Compute Cm per direction + stats. Same disk-first contract as run_cf.

    Per body and cfg label:

    - ``{output}/Cm.{label}.{body}.time_series.h5`` + ``.xdmf``
    - ``stats.h5``: ``/cm_{dir}/{label}/{body}/{Triangles, Geometry,
      stat...}``

    Multi-case moment centers: when a ``MomentBodyConfig`` declares
    ``lever_origin_cases`` or ``lever_strategy="region_bbox_corners_xy"``,
    each case is expanded into a separate derived body (named
    ``{body}.{case_label}``). Every case produces its own timeseries file
    and stats group, computed independently, so callers can scan candidate
    moment centers and pick the worst-case afterwards.

    ``mesh_path`` accepts ``.lnas``, ``.stl``, ``.h5``, ``.xdmf``, or a
    pre-loaded :class:`LnasFormat`. If omitted, the geometry is read from
    ``cp_h5`` itself (single ``"all"`` surface).
    """
    case_cfg = _coerce_case_cfg(cfg_path, CmCaseConfig)
    _inherit_full_scale_from_cp(case_cfg, cp_h5)
    mesh = mesh_from_h5(cp_h5) if mesh_path is None else load_mesh(mesh_path)
    path_manager = CmPathManager(output_path=output)
    stats_h5 = path_manager.get_stats_h5_path()
    create_folders_for_file(stats_h5)

    for cfg_lbl, cfg in case_cfg.moment_coefficient.items():
        logger.info(f"Processing Cm: {cfg_lbl}")
        runs = _expand_moment_cases(cfg, case_cfg.bodies, mesh)
        if len(runs) != len(cfg.bodies):
            logger.info(
                f"Cm cases: expanded {len(cfg.bodies)} body(ies) into "
                f"{len(runs)} independent runs"
            )
        any_stats = False
        for run_cfg, run_bodies_def in runs:
            _run_body_coefficient(
                coef="cm",
                mesh=mesh,
                cp_h5=cp_h5,
                cfg_lbl=cfg_lbl,
                cfg=run_cfg,
                bodies_definition=run_bodies_def,
                process_fn=process_Cm,
                path_manager=path_manager,
                stats_h5=stats_h5,
            )
            any_stats = any_stats or bool(run_cfg.statistics)
        if any_stats and stats_h5.exists():
            write_stats_xdmf(stats_h5, path_manager.get_stats_xdmf_path())
            logger.info(f"Cm stats written for config '{cfg_lbl}'")
        else:
            logger.info(f"Cm '{cfg_lbl}': statistics list is empty; skipping stats step")


def run_ce(
    cp_h5: pathlib.Path,
    cfg_path: pathlib.Path | str | CpCaseConfig | CfCaseConfig | CmCaseConfig | CeCaseConfig,
    output: pathlib.Path,
    mesh_path: pathlib.Path | LnasFormat | None = None,
) -> None:
    """Compute Ce + stats.

    Ce slices triangles along zoning planes via process_surfaces; the resulting
    cut regions mesh is written to disk alongside the per-cut-triangle
    timeseries and stats.

    Per cfg label:

    - ``{output}/Ce.{label}.time_series.h5`` + ``.xdmf`` -- root
      ``/Triangles`` + ``/Geometry`` is the cut mesh; ``/ce/t{T}`` per
      timestep.
    - ``{output}/Ce.{label}.regions.stl`` -- cut mesh as STL for
      QC/ParaView.
    - ``stats.h5``: ``/ce/{label}/{Triangles, Geometry, stat...}``.

    ``mesh_path`` accepts ``.lnas``, ``.stl``, ``.h5``, ``.xdmf``, or a
    pre-loaded :class:`LnasFormat`. If omitted, the geometry is read from
    ``cp_h5`` itself (single ``"all"`` surface).
    """
    case_cfg = _coerce_case_cfg(cfg_path, CeCaseConfig)
    _inherit_full_scale_from_cp(case_cfg, cp_h5)
    mesh = mesh_from_h5(cp_h5) if mesh_path is None else load_mesh(mesh_path)
    path_manager = CePathManager(output_path=output)
    stats_h5 = path_manager.get_stats_h5_path()
    create_folders_for_file(stats_h5)

    for cfg_lbl, cfg in case_cfg.shape_coefficient.items():
        logger.info(f"Processing Ce: {cfg_lbl}")

        ce_output = process_Ce(mesh=mesh, cfg=cfg, cp_h5=cp_h5)

        # Build the unified cut mesh and per-cut-tri region label array.
        cut_mesh = ce_output.processed_entities[0].mesh.copy()
        if len(ce_output.processed_entities) > 1:
            cut_mesh.join([entity.mesh.copy() for entity in ce_output.processed_entities[1:]])
        cut_region_idx = ce_output.region_indexing_df["region_idx"].to_numpy()

        # Persist cut mesh as STL for ParaView/QC.
        regions_stl = path_manager.get_regions_stl_path(cfg_lbl=cfg_lbl)
        create_folders_for_file(regions_stl)
        cut_mesh.export_stl(regions_stl)

        # Write timeseries with cut mesh + broadcast per-region values.
        ts_path = path_manager.get_timeseries_path(cfg_lbl=cfg_lbl)
        create_folders_for_file(ts_path)
        if ts_path.exists():
            ts_path.unlink()

        write_timeseries_geometry(ts_path, cut_mesh.triangles, cut_mesh.vertices)
        times = _write_region_timeseries(
            ts_path,
            triangles=cut_mesh.triangles,
            vertices=cut_mesh.vertices,
            region_idx_per_tri=cut_region_idx,
            data_df=ce_output.data_df,
            group="ce",
        )
        write_timeseries_meta(ts_path, _cp_time_steps_for(cp_h5, times), times)
        write_temporal_xdmf(ts_path, ts_path.with_suffix(".xdmf"), group="ce")

        cfg_dump = cfg.model_dump()
        meta_extra = {
            "coefficient": "ce",
            "cfg_lbl": cfg_lbl,
            "cp_h5": str(cp_h5),
            "regions_stl": str(regions_stl),
        }
        write_processing_metadata(ts_path, "/", cfg_dump, extra=meta_extra)

        if cfg.statistics:
            _write_stats_for_group(
                stats_h5,
                timeseries_path=ts_path,
                timeseries_group="ce",
                stats_group=f"ce/{cfg_lbl}",
                statistics=cfg.statistics,
                triangles=cut_mesh.triangles,
                vertices=cut_mesh.vertices,
            )
            write_processing_metadata(stats_h5, f"ce/{cfg_lbl}", cfg_dump, extra=meta_extra)

            write_stats_xdmf(stats_h5, path_manager.get_stats_xdmf_path())
            logger.info(f"Ce stats written for config '{cfg_lbl}'")
        else:
            logger.info(f"Ce '{cfg_lbl}': statistics list is empty; skipping stats step")
