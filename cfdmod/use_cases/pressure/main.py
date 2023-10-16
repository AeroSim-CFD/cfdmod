import argparse
from dataclasses import dataclass

import pandas as pd
from nassu.lnas import LagrangianFormat

from cfdmod.api.vtk.write_vtk import create_polydata_for_cell_data, write_polydata
from cfdmod.logger import logger
from cfdmod.use_cases.pressure.cp_config import CpConfig
from cfdmod.use_cases.pressure.cp_data import (
    calculate_statistics,
    filter_pressure_data,
    transform_to_cp,
)
from cfdmod.use_cases.pressure.path_manager import CpPathManager


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
    path_manager = CpPathManager(
        args_use.output, args_use.config, args_use.mesh, args_use.p, args_use.s
    )
    post_proc_cfg = CpConfig.from_file(path_manager.config_path)
    logger.info("Reading mesh description...")
    mesh = LagrangianFormat.from_file(path_manager.mesh_path)
    logger.info("Mesh description loaded successfully!")

    logger.info("Preparing to read pressure data...")
    press_data: pd.DataFrame = pd.read_hdf(path_manager.static_data_path)  # type: ignore
    body_data: pd.DataFrame = pd.read_hdf(path_manager.body_data_path)  # type: ignore
    press_data, body_data = filter_pressure_data(
        press_data, body_data, post_proc_cfg.timestep_range
    )
    logger.info("Read pressure data successfully!")

    # OUTPUT 1: cp(t)
    cp_data = transform_to_cp(
        press_data,
        body_data,
        reference_vel=post_proc_cfg.U_H,
        ref_press_mode=post_proc_cfg.reference_pressure,
    )
    logger.info("Transformed pressure into coefficients")
    cp_data.to_hdf(path_manager.cp_t_path, key="cp_t", mode="w", index=False)
    logger.info("Exported coefficients")

    # OUTPUT 2: cp_stats
    cp_stats = calculate_statistics(cp_data, statistics_to_apply=post_proc_cfg.statistics)
    cp_stats.to_hdf(path_manager.cp_stats_path, key="cp_t", mode="w", index=False)
    logger.info("Exported statistics")

    polydata = create_polydata_for_cell_data(data=cp_stats, mesh=mesh.geometry)

    # OUTPUT 3: VTK cp_stats
    write_polydata(path_manager.vtp_path, polydata)
    logger.info("Exported VTK file")
