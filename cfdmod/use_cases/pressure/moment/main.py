import argparse
import pathlib
from dataclasses import dataclass

import pandas as pd
from nassu.lnas import LagrangianFormat

from cfdmod.logger import logger
from cfdmod.use_cases.pressure.moment.Cm_config import CmCaseConfig
from cfdmod.use_cases.pressure.moment.Cm_data import process_body
from cfdmod.use_cases.pressure.path_manager import CmPathManager


@dataclass
class ArgsModel:
    """Command line arguments for client app"""

    output: str
    cp: str
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
        "--cp",
        required=True,
        help="Path to body pressure coefficient series .hdf",
        type=str,
    )
    ap.add_argument(
        "--mesh",
        required=True,
        help="Path to LNAS normalized mesh file",
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
    path_manager = CmPathManager(output_path=pathlib.Path(args_use.output))

    cfg_path = pathlib.Path(args_use.config)
    mesh_path = pathlib.Path(args_use.mesh)
    cp_path = pathlib.Path(args_use.cp)

    post_proc_cfg = CmCaseConfig.from_file(cfg_path)

    logger.info("Reading mesh description...")
    mesh = LagrangianFormat.from_file(mesh_path)
    logger.info("Mesh description loaded successfully!")

    logger.info("Preparing to read pressure coefficients data...")
    cp_data = pd.read_hdf(cp_path)

    cp_data_to_use = cp_data.to_frame() if isinstance(cp_data, pd.Series) else cp_data
    logger.info("Read pressure coefficient data successfully!")

    vec_areas = mesh.geometry._cross_prod() / 2
    areas_df = pd.DataFrame({"Ax": vec_areas[:, 0], "Ay": vec_areas[:, 1], "Az": vec_areas[:, 2]})
    areas_df["point_idx"] = areas_df.index

    cp_data = pd.merge(cp_data_to_use, areas_df, on="point_idx", how="left")

    for cfg_label, cfg in post_proc_cfg.moment_coefficient.items():
        for body_label in cfg.bodies:
            body_cfg = post_proc_cfg.bodies[body_label]
            logger.info(f"Processing body {body_label} ...")

            processed_body = process_body(mesh=mesh, body_cfg=body_cfg, cp_data=cp_data, cfg=cfg)
            processed_body.save_outputs(
                body_label=body_label, cfg_label=cfg_label, path_manager=path_manager
            )

            logger.info(f"Processed body {body_label}")
