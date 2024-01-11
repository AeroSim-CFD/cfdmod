import argparse
import pathlib
from dataclasses import dataclass

import pandas as pd
from lnas import LnasFormat

from cfdmod.api.vtk.write_vtk import create_polydata_for_cell_data, write_polydata
from cfdmod.logger import logger
from cfdmod.use_cases.pressure.chunking import split_into_chunks
from cfdmod.use_cases.pressure.cp_config import CpCaseConfig
from cfdmod.use_cases.pressure.cp_data import filter_pressure_data, transform_to_cp
from cfdmod.use_cases.pressure.path_manager import CpPathManager, copy_input_artifacts
from cfdmod.use_cases.pressure.zoning.processing import calculate_statistics
from cfdmod.utils import create_folders_for_file


@dataclass
class ArgsModel:
    """Command line arguments for client app"""

    output: str
    p: str
    s: str
    mesh: str
    config: str


def get_args_process(args: list[str]) -> ArgsModel:
    """Get arguments model from list of command line args

    Args:
        args (List[str]): List of command line arguments passed

    Returns:
        ArgsModel: Arguments model for client app
    """
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--output",
        required=True,
        help="Output path for generated files",
        type=str,
    )
    ap.add_argument(
        "--p",
        required=True,
        help="Path to body pressure series .hdf",
        type=str,
    )
    ap.add_argument(
        "--s",
        required=True,
        help="Path to static pressure series .hdf",
        type=str,
    )
    ap.add_argument(
        "--mesh",
        required=True,
        help="Path to LNAS mesh file",
        type=str,
    )
    ap.add_argument(
        "--config",
        required=True,
        help="Path to config .yaml file",
        type=str,
    )
    parsed_args = ap.parse_args(args)
    args_model = ArgsModel(**vars(parsed_args))
    return args_model


def main(*args):
    args_use = get_args_process(*args)
    path_manager = CpPathManager(output_path=pathlib.Path(args_use.output))

    cfg_path = pathlib.Path(args_use.config)
    mesh_path = pathlib.Path(args_use.mesh)
    static_data_path = pathlib.Path(args_use.s)
    body_data_path = pathlib.Path(args_use.p)

    copy_input_artifacts(
        cfg_path=cfg_path,
        mesh_path=mesh_path,
        static_data_path=static_data_path,
        body_data_path=body_data_path,
        path_manager=path_manager,
    )

    post_proc_cfg = CpCaseConfig.from_file(cfg_path)

    logger.info("Reading mesh description...")
    mesh = LnasFormat.from_file(mesh_path)
    logger.info("Mesh description loaded successfully!")

    logger.info("Preparing to read pressure data...")
    pressure_data: pd.DataFrame = pd.read_hdf(static_data_path)  # type: ignore
    original_body_data: pd.DataFrame = pd.read_hdf(body_data_path)  # type: ignore
    logger.info("Read pressure data successfully!")

    for cfg_lbl, cfg in post_proc_cfg.pressure_coefficient.items():
        logger.info(f"Processing pressure coefficients for config {cfg_lbl} ...")
        press_data, body_data = filter_pressure_data(
            pressure_data, original_body_data, cfg.timestep_range
        )

        # OUTPUT 1: cp(t)
        cp_data = transform_to_cp(
            press_data,
            body_data,
            reference_vel=cfg.U_H,
            ref_press_mode=cfg.reference_pressure,
            correction_factor=cfg.U_H_correction_factor,
        )
        logger.info("Transformed pressure into coefficients")

        timeseries_path = path_manager.get_cp_t_path(cfg_label=cfg_lbl)
        create_folders_for_file(timeseries_path)

        split_into_chunks(
            time_series_df=cp_data,
            number_of_chunks=cfg.number_of_chunks,
            output_path=timeseries_path,
        )
        logger.info("Exported coefficients")

        # OUTPUT 2: cp_stats
        cp_stats = calculate_statistics(
            cp_data,
            statistics_to_apply=cfg.statistics,
            variables=["cp"],
            group_by_key="point_idx",
            extreme_params=post_proc_cfg.extreme_values,
        )
        stats_path = path_manager.get_cp_stats_path(cfg_label=cfg_lbl)
        create_folders_for_file(stats_path)
        cp_stats.to_hdf(stats_path, key="cp_stats", mode="w", index=False)
        logger.info("Exported statistics")

        polydata = create_polydata_for_cell_data(data=cp_stats, mesh=mesh.geometry)

        # OUTPUT 3: VTK cp_stats
        vtp_path = path_manager.get_vtp_path(cfg_label=cfg_lbl)
        create_folders_for_file(vtp_path)
        write_polydata(vtp_path, polydata)
        logger.info("Exported VTK file")
