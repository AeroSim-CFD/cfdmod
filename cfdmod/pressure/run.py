"""Orchestration functions for the pressure module.

Pure Python entry points called by cli.py. No argparse or file-path logic here.

Pipeline contract: every coefficient first persists its full per-triangle
timeseries to disk (XDMF+H5), then computes statistics from that on-disk
file via cfdmod.pressure.statistics_runner.calculate_statistics_from_h5.
Stats are appended to a single combined results.h5 with an embedded mesh
per leaf group (so different sub-meshes — body subsets for Cf/Cm, sliced
regions mesh for Ce — coexist without length collisions).
"""

from __future__ import annotations

__all__ = ["run_cp", "run_cf", "run_cm", "run_ce"]

import pathlib

import h5py
import numpy as np
import pandas as pd
from lnas import LnasFormat

from cfdmod.io.xdmf import (
    write_stats_field,
    write_stats_xdmf,
    write_temporal_xdmf,
    write_timeseries_geometry,
    write_timeseries_meta,
    write_timeseries_step,
)
from cfdmod.logger import logger
from cfdmod.pressure.functions import (
    process_Ce,
    process_Cf,
    process_Cm,
    process_xdmf_to_cp,
)
from cfdmod.pressure.parameters import CeCaseConfig, CfCaseConfig, CmCaseConfig, CpCaseConfig
from cfdmod.pressure.path_manager import CePathManager, CfPathManager, CmPathManager, CpPathManager
from cfdmod.pressure.statistics_runner import calculate_statistics_from_h5
from cfdmod.utils import create_folders_for_file, save_yaml


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
    # with NaN — same behaviour as process_surfaces' existing left-join warn.
    col_pos = {c: i for i, c in enumerate(data_df.columns)}
    tri_col_idx = np.array(
        [col_pos.get(r, -1) for r in region_idx_per_tri], dtype=np.int64
    )
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
    results_h5: pathlib.Path,
    *,
    timeseries_path: pathlib.Path,
    timeseries_group: str,
    stats_group: str,
    statistics,
    triangles: np.ndarray,
    vertices: np.ndarray,
) -> None:
    """Compute stats from an on-disk timeseries group and append to results.h5
    under ``stats_group`` with the embedded mesh."""
    stats_df = calculate_statistics_from_h5(
        h5_path=timeseries_path,
        group=timeseries_group,
        statistics=statistics,
        timestep_range=None,
    )
    for stat_name in stats_df.columns:
        write_stats_field(
            h5_path=results_h5,
            group=stats_group,
            stat_name=stat_name,
            values=stats_df[stat_name].to_numpy(dtype=np.float64),
            triangles=triangles,
            vertices=vertices,
        )


def run_cp(
    body_h5: pathlib.Path,
    probe_h5: pathlib.Path | None,
    mesh_path: pathlib.Path,
    cfg_path: pathlib.Path,
    output: pathlib.Path,
) -> None:
    """Compute Cp timeseries + stats.

    Outputs per config label:
      - {output}/cp/{label}/cp.time_series.h5 + .xdmf      (timeseries)
      - {output}/results.h5 + results.xdmf                  (stats; embeds the
        full mesh under /cp/{label}/ alongside stat datasets)
    """
    case_cfg = CpCaseConfig.from_file(cfg_path)
    mesh = LnasFormat.from_file(mesh_path)
    path_manager = CpPathManager(output_path=output)
    results_h5 = path_manager.get_results_h5_path()
    create_folders_for_file(results_h5)

    triangles = mesh.geometry.triangles
    vertices = mesh.geometry.vertices

    for cfg_lbl, cfg in case_cfg.pressure_coefficient.items():
        logger.info(f"Processing Cp: {cfg_lbl}")

        timeseries_path = path_manager.get_timeseries_path(cfg_lbl=cfg_lbl)
        create_folders_for_file(timeseries_path)

        config_path = path_manager.get_config_path(cfg_lbl=cfg_lbl)
        create_folders_for_file(config_path)
        save_yaml(cfg.model_dump(), config_path)

        logger.info("Transforming to Cp timeseries...")
        process_xdmf_to_cp(
            body_h5=body_h5,
            probe_h5=probe_h5,
            output_path=timeseries_path,
            cp_config=cfg,
        )

        logger.info("Calculating Cp statistics from on-disk timeseries...")
        _write_stats_for_group(
            results_h5,
            timeseries_path=timeseries_path,
            timeseries_group="cp",
            stats_group=f"cp/{cfg_lbl}",
            statistics=cfg.statistics,
            triangles=triangles,
            vertices=vertices,
        )

        write_stats_xdmf(results_h5, path_manager.get_results_xdmf_path())
        logger.info(f"Cp stats written for config '{cfg_lbl}'")


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
    results_h5: pathlib.Path,
) -> None:
    """Shared body-coefficient flow for Cf and Cm.

    1. Run the in-memory transform (process_Cf or process_Cm) to get per-region
       Cf/Cm matrices keyed by direction.
    2. For each body: write a per-body timeseries H5 with one group per
       direction (cf_x/cf_y/cf_z), broadcasting the per-region values onto
       body-mesh triangles via the region-indexing df produced upstream.
    3. Compute stats from the on-disk file and append to results.h5 under
       /{coef}_{dir}/{cfg_lbl}/{body}/ with the body mesh embedded.
    """
    compiled_output = process_fn(
        mesh=mesh,
        cfg=cfg,
        cp_h5=cp_h5,
        bodies_definition=bodies_definition,
    )

    config_path = path_manager.get_config_path(cfg_lbl=cfg_lbl)
    create_folders_for_file(config_path)
    save_yaml(cfg.model_dump(), config_path)

    geometry_df = compiled_output[cfg.directions[0]].region_indexing_df

    for body_cfg in cfg.bodies:
        body_geom = mesh.geometry_from_list_surfaces(
            surfaces_names=bodies_definition[body_cfg.name].surfaces
        )[0]
        body_region_idx = geometry_df.loc[
            geometry_df.region_idx.str.contains(body_cfg.name)
        ].region_idx.to_numpy()

        ts_path = path_manager.get_body_timeseries_path(
            cfg_lbl=cfg_lbl, body_name=body_cfg.name
        )
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
        write_timeseries_meta(ts_path, times, times)
        write_temporal_xdmf(
            ts_path,
            ts_path.with_suffix(".xdmf"),
            group=[f"{coef}_{d}" for d in cfg.directions],
        )

        for direction in cfg.directions:
            _write_stats_for_group(
                results_h5,
                timeseries_path=ts_path,
                timeseries_group=f"{coef}_{direction}",
                stats_group=f"{coef}_{direction}/{cfg_lbl}/{body_cfg.name}",
                statistics=cfg.statistics,
                triangles=body_geom.triangles,
                vertices=body_geom.vertices,
            )


def run_cf(
    cp_h5: pathlib.Path,
    mesh_path: pathlib.Path,
    cfg_path: pathlib.Path,
    output: pathlib.Path,
) -> None:
    """Compute Cf per direction + stats.

    Per body & cfg label:
      - {output}/Cf/{label}/{body}.time_series.h5 + .xdmf     (one group per
        direction: /cf_x, /cf_y, /cf_z, each with /t{T} arrays of length
        n_body_tri)
      - results.h5: /cf_{dir}/{label}/{body}/{Triangles, Geometry, stat...}
    """
    case_cfg = CfCaseConfig.from_file(cfg_path)
    mesh = LnasFormat.from_file(mesh_path)
    path_manager = CfPathManager(output_path=output)
    results_h5 = path_manager.get_results_h5_path()
    create_folders_for_file(results_h5)

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
            results_h5=results_h5,
        )
        write_stats_xdmf(results_h5, path_manager.get_results_xdmf_path())
        logger.info(f"Cf stats written for config '{cfg_lbl}'")


def run_cm(
    cp_h5: pathlib.Path,
    mesh_path: pathlib.Path,
    cfg_path: pathlib.Path,
    output: pathlib.Path,
) -> None:
    """Compute Cm per direction + stats. Same disk-first contract as run_cf.

    Per body & cfg label:
      - {output}/Cm/{label}/{body}.time_series.h5 + .xdmf
      - results.h5: /cm_{dir}/{label}/{body}/{Triangles, Geometry, stat...}
    """
    case_cfg = CmCaseConfig.from_file(cfg_path)
    mesh = LnasFormat.from_file(mesh_path)
    path_manager = CmPathManager(output_path=output)
    results_h5 = path_manager.get_results_h5_path()
    create_folders_for_file(results_h5)

    for cfg_lbl, cfg in case_cfg.moment_coefficient.items():
        logger.info(f"Processing Cm: {cfg_lbl}")
        _run_body_coefficient(
            coef="cm",
            mesh=mesh,
            cp_h5=cp_h5,
            cfg_lbl=cfg_lbl,
            cfg=cfg,
            bodies_definition=case_cfg.bodies,
            process_fn=process_Cm,
            path_manager=path_manager,
            results_h5=results_h5,
        )
        write_stats_xdmf(results_h5, path_manager.get_results_xdmf_path())
        logger.info(f"Cm stats written for config '{cfg_lbl}'")


def run_ce(
    cp_h5: pathlib.Path,
    mesh_path: pathlib.Path,
    cfg_path: pathlib.Path,
    output: pathlib.Path,
) -> None:
    """Compute Ce + stats.

    Ce slices triangles along zoning planes via process_surfaces; the resulting
    cut regions mesh is written to disk alongside the per-cut-triangle
    timeseries and stats.

    Per cfg label:
      - {output}/Ce/{label}/Ce.time_series.h5 + .xdmf
        (root /Triangles+/Geometry = cut mesh; /ce/t{T} per timestep)
      - {output}/Ce/{label}/regions.stl    (cut mesh as STL for QC/ParaView)
      - results.h5: /ce/{label}/{Triangles, Geometry, stat...}
    """
    case_cfg = CeCaseConfig.from_file(cfg_path)
    mesh = LnasFormat.from_file(mesh_path)
    path_manager = CePathManager(output_path=output)
    results_h5 = path_manager.get_results_h5_path()
    create_folders_for_file(results_h5)

    for cfg_lbl, cfg in case_cfg.shape_coefficient.items():
        logger.info(f"Processing Ce: {cfg_lbl}")

        config_path = path_manager.get_config_path(cfg_lbl=cfg_lbl)
        create_folders_for_file(config_path)
        save_yaml(cfg.model_dump(), config_path)

        ce_output = process_Ce(mesh=mesh, cfg=cfg, cp_h5=cp_h5)

        # Build the unified cut mesh and per-cut-tri region label array.
        cut_mesh = ce_output.processed_entities[0].mesh.copy()
        if len(ce_output.processed_entities) > 1:
            cut_mesh.join(
                [entity.mesh.copy() for entity in ce_output.processed_entities[1:]]
            )
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
        write_timeseries_meta(ts_path, times, times)
        write_temporal_xdmf(ts_path, ts_path.with_suffix(".xdmf"), group="ce")

        _write_stats_for_group(
            results_h5,
            timeseries_path=ts_path,
            timeseries_group="ce",
            stats_group=f"ce/{cfg_lbl}",
            statistics=cfg.statistics,
            triangles=cut_mesh.triangles,
            vertices=cut_mesh.vertices,
        )

        write_stats_xdmf(results_h5, path_manager.get_results_xdmf_path())
        logger.info(f"Ce stats written for config '{cfg_lbl}'")
