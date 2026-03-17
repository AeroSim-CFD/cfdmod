"""Orchestration functions for the pressure module.

Pure Python entry points called by cli.py. No argparse or file-path logic here.
"""

from __future__ import annotations

__all__ = ["run_cp", "run_cf", "run_cm", "run_ce"]

import pathlib

from lnas import LnasFormat

from cfdmod.io.xdmf import write_stats_field, write_stats_xdmf
from cfdmod.logger import logger
from cfdmod.pressure.functions import (
    combine_stats_data_with_mesh,
    process_Ce,
    process_Cf,
    process_Cm,
    process_xdmf_to_cp,
)
from cfdmod.pressure.parameters import CeCaseConfig, CfCaseConfig, CmCaseConfig, CpCaseConfig
from cfdmod.pressure.path_manager import CePathManager, CfPathManager, CmPathManager, CpPathManager
from cfdmod.pressure.statistics_runner import calculate_statistics_from_h5
from cfdmod.utils import create_folders_for_file, save_yaml


def run_cp(
    body_h5: pathlib.Path,
    probe_h5: pathlib.Path | None,
    mesh_path: pathlib.Path,
    cfg_path: pathlib.Path,
    output: pathlib.Path,
) -> None:
    """Compute Cp timeseries + stats.

    Outputs per config label:
      - {output}/cp/{label}/cp.time_series.h5 + .xdmf    (timeseries)
      - {output}/results.h5 + results.xdmf               (stats, /cp/{stat} fields added)

    Args:
        body_h5 (pathlib.Path): Body pressure H5 (pressure/t{T} per timestep)
        probe_h5 (pathlib.Path | None): Atmospheric probe H5
        mesh_path (pathlib.Path): LNAS mesh file
        cfg_path (pathlib.Path): Cp YAML config file
        output (pathlib.Path): Output directory
    """
    case_cfg = CpCaseConfig.from_file(cfg_path)
    mesh = LnasFormat.load(mesh_path)
    path_manager = CpPathManager(output_path=output)

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

        logger.info("Calculating Cp statistics...")
        cp_stats = calculate_statistics_from_h5(
            h5_path=timeseries_path,
            group="cp",
            statistics=cfg.statistics,
            timestep_range=None,
        )

        results_h5 = path_manager.get_results_h5_path()
        create_folders_for_file(results_h5)

        triangles = mesh.geometry.triangles
        vertices = mesh.geometry.vertices

        for stat_name in cp_stats.columns:
            values = cp_stats[stat_name].to_numpy()
            write_stats_field(
                h5_path=results_h5,
                group=f"cp/{cfg_lbl}",
                stat_name=stat_name,
                values=values,
                triangles=triangles,
                vertices=vertices,
            )

        results_xdmf = path_manager.get_results_xdmf_path()
        write_stats_xdmf(results_h5, results_xdmf)
        logger.info(f"Cp stats written for config '{cfg_lbl}'")


def run_cf(
    cp_h5: pathlib.Path,
    mesh_path: pathlib.Path,
    cfg_path: pathlib.Path,
    output: pathlib.Path,
) -> None:
    """Compute Cf per direction + stats.

    Adds /cf_{dir}/{cfg_lbl}/{stat} fields to results.h5, regenerates results.xdmf.

    Args:
        cp_h5 (pathlib.Path): Cp timeseries H5
        mesh_path (pathlib.Path): LNAS mesh file
        cfg_path (pathlib.Path): Cf YAML config file
        output (pathlib.Path): Output directory
    """
    case_cfg = CfCaseConfig.from_file(cfg_path)
    mesh = LnasFormat.load(mesh_path)
    path_manager = CfPathManager(output_path=output)

    results_h5 = path_manager.get_results_h5_path()
    create_folders_for_file(results_h5)

    triangles = mesh.geometry.triangles
    vertices = mesh.geometry.vertices

    for cfg_lbl, cfg in case_cfg.force_coefficient.items():
        logger.info(f"Processing Cf: {cfg_lbl}")

        config_path = path_manager.get_config_path(cfg_lbl=cfg_lbl)
        create_folders_for_file(config_path)
        save_yaml(cfg.model_dump(), config_path)

        compiled_output = process_Cf(
            mesh=mesh,
            cfg=cfg,
            cp_h5=cp_h5,
            bodies_definition=case_cfg.bodies,
        )

        for direction_lbl, output_data in compiled_output.items():
            stats_df = output_data.stats_df
            geometry_df = output_data.region_indexing_df

            for body_cfg in cfg.bodies:
                body_region_idx = geometry_df.loc[
                    geometry_df.region_idx.str.contains(body_cfg.name)
                ].region_idx.to_numpy()

                from cfdmod.pressure.geometry import GeometryData, get_indexing_mask
                from cfdmod.pressure.parameters import ZoningModel
                import numpy as np

                body_geom = mesh.geometry_from_list_surfaces(
                    surfaces_names=case_cfg.bodies[body_cfg.name].surfaces
                )[0]
                tri_stats = combine_stats_data_with_mesh(
                    mesh=body_geom,
                    region_idx_array=body_region_idx,
                    data_stats=stats_df,
                )

                for stat_name in stats_df.columns:
                    write_stats_field(
                        h5_path=results_h5,
                        group=f"cf_{direction_lbl}/{cfg_lbl}/{body_cfg.name}",
                        stat_name=stat_name,
                        values=tri_stats[stat_name].to_numpy(),
                        triangles=triangles,
                        vertices=vertices,
                    )

        results_xdmf = path_manager.get_results_xdmf_path()
        write_stats_xdmf(results_h5, results_xdmf)
        logger.info(f"Cf stats written for config '{cfg_lbl}'")


def run_cm(
    cp_h5: pathlib.Path,
    mesh_path: pathlib.Path,
    cfg_path: pathlib.Path,
    output: pathlib.Path,
) -> None:
    """Compute Cm per direction + stats.

    Adds /cm_{dir}/{cfg_lbl}/{stat} fields to results.h5, regenerates results.xdmf.

    Args:
        cp_h5 (pathlib.Path): Cp timeseries H5
        mesh_path (pathlib.Path): LNAS mesh file
        cfg_path (pathlib.Path): Cm YAML config file
        output (pathlib.Path): Output directory
    """
    case_cfg = CmCaseConfig.from_file(cfg_path)
    mesh = LnasFormat.load(mesh_path)
    path_manager = CmPathManager(output_path=output)

    results_h5 = path_manager.get_results_h5_path()
    create_folders_for_file(results_h5)

    triangles = mesh.geometry.triangles
    vertices = mesh.geometry.vertices

    for cfg_lbl, cfg in case_cfg.moment_coefficient.items():
        logger.info(f"Processing Cm: {cfg_lbl}")

        config_path = path_manager.get_config_path(cfg_lbl=cfg_lbl)
        create_folders_for_file(config_path)
        save_yaml(cfg.model_dump(), config_path)

        compiled_output = process_Cm(
            mesh=mesh,
            cfg=cfg,
            cp_h5=cp_h5,
            bodies_definition=case_cfg.bodies,
        )

        for direction_lbl, output_data in compiled_output.items():
            stats_df = output_data.stats_df
            geometry_df = output_data.region_indexing_df

            for body_cfg in cfg.bodies:
                body_region_idx = geometry_df.loc[
                    geometry_df.region_idx.str.contains(body_cfg.name)
                ].region_idx.to_numpy()

                body_geom = mesh.geometry_from_list_surfaces(
                    surfaces_names=case_cfg.bodies[body_cfg.name].surfaces
                )[0]
                tri_stats = combine_stats_data_with_mesh(
                    mesh=body_geom,
                    region_idx_array=body_region_idx,
                    data_stats=stats_df,
                )

                for stat_name in stats_df.columns:
                    write_stats_field(
                        h5_path=results_h5,
                        group=f"cm_{direction_lbl}/{cfg_lbl}/{body_cfg.name}",
                        stat_name=stat_name,
                        values=tri_stats[stat_name].to_numpy(),
                        triangles=triangles,
                        vertices=vertices,
                    )

        results_xdmf = path_manager.get_results_xdmf_path()
        write_stats_xdmf(results_h5, results_xdmf)
        logger.info(f"Cm stats written for config '{cfg_lbl}'")


def run_ce(
    cp_h5: pathlib.Path,
    mesh_path: pathlib.Path,
    cfg_path: pathlib.Path,
    output: pathlib.Path,
) -> None:
    """Compute Ce + stats.

    Adds /ce/{cfg_lbl}/{stat} fields to results.h5, regenerates results.xdmf.

    Args:
        cp_h5 (pathlib.Path): Cp timeseries H5
        mesh_path (pathlib.Path): LNAS mesh file
        cfg_path (pathlib.Path): Ce YAML config file
        output (pathlib.Path): Output directory
    """
    case_cfg = CeCaseConfig.from_file(cfg_path)
    mesh = LnasFormat.load(mesh_path)
    path_manager = CePathManager(output_path=output)

    results_h5 = path_manager.get_results_h5_path()
    create_folders_for_file(results_h5)

    triangles = mesh.geometry.triangles
    vertices = mesh.geometry.vertices

    for cfg_lbl, cfg in case_cfg.shape_coefficient.items():
        logger.info(f"Processing Ce: {cfg_lbl}")

        config_path = path_manager.get_config_path(cfg_lbl=cfg_lbl)
        create_folders_for_file(config_path)
        save_yaml(cfg.model_dump(), config_path)

        ce_output = process_Ce(
            mesh=mesh,
            cfg=cfg,
            cp_h5=cp_h5,
        )

        stats_df = ce_output.stats_df
        for stat_name in stats_df.columns:
            write_stats_field(
                h5_path=results_h5,
                group=f"ce/{cfg_lbl}",
                stat_name=stat_name,
                values=stats_df[stat_name].to_numpy(),
                triangles=triangles,
                vertices=vertices,
            )

        results_xdmf = path_manager.get_results_xdmf_path()
        write_stats_xdmf(results_h5, results_xdmf)
        logger.info(f"Ce stats written for config '{cfg_lbl}'")
