import argparse
import pathlib
from dataclasses import dataclass

import pandas as pd
from lnas import LnasFormat

from cfdmod.logger import logger
from cfdmod.use_cases.pressure.cp_config import CpCaseConfig
from cfdmod.use_cases.pressure.cp_data import process_cp
from cfdmod.use_cases.pressure.path_manager import CpPathManager, copy_input_artifacts


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
        help="Path to body pressure series .h5",
        type=str,
    )
    ap.add_argument(
        "--s",
        required=True,
        help="Path to static pressure series .h5",
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
    body_data: pd.DataFrame = pd.read_hdf(body_data_path)  # type: ignore
    logger.info("Read pressure data successfully!")

    for cfg_lbl, cfg in post_proc_cfg.pressure_coefficient.items():
        logger.info(f"Processing pressure coefficients for config {cfg_lbl} ...")

        cp_output = process_cp(
            pressure_data=pressure_data,
            body_data=body_data,
            cfg=cfg,
            mesh=mesh.geometry,
            extreme_params=post_proc_cfg.extreme_values,
        )
        cp_output.save_outputs(cfg=cfg, cfg_label=cfg_lbl, path_manager=path_manager)

        logger.info(f"Processed pressure coefficients for config {cfg_lbl}!")
