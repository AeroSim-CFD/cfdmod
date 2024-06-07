import argparse
import pathlib
from dataclasses import dataclass

from lnas import LnasFormat

from cfdmod.logger import logger
from cfdmod.use_cases.pressure.force.Cf_config import CfCaseConfig
from cfdmod.use_cases.pressure.force.Cf_data import process_Cf
from cfdmod.use_cases.pressure.output import CommonOutput
from cfdmod.use_cases.pressure.path_manager import CfPathManager
from cfdmod.utils import save_yaml


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

    cfg_path = pathlib.Path(args_use.config)
    mesh_path = pathlib.Path(args_use.mesh)
    cp_path = pathlib.Path(args_use.cp)

    post_proc_cfg = CfCaseConfig.from_file(cfg_path)
    path_manager = CfPathManager(output_path=pathlib.Path(args_use.output))

    logger.info("Reading mesh description...")
    mesh = LnasFormat.from_file(mesh_path)
    logger.info("Mesh description loaded successfully!")

    for cfg_label, cfg in post_proc_cfg.force_coefficient.items():
        logger.info(f"Processing Cf config {cfg_label} ...")

        cf_output_dict: dict[str, CommonOutput] = process_Cf(
            mesh=mesh, cfg=cfg, cp_path=cp_path, bodies_definition=post_proc_cfg.bodies
        )
        already_saved = False
        for direction_lbl, cf_output in cf_output_dict.items():
            path_manager.direction_label = direction_lbl
            if already_saved:
                cf_output.save_outputs(cfg_label=cfg_label, path_manager=path_manager)
            else:
                cf_output.save_region_info(cfg_label=cfg_label, path_manager=path_manager)
                cf_output.save_outputs(cfg_label=cfg_label, path_manager=path_manager)
                already_saved = True
            save_yaml(cfg.model_dump(), path_manager.get_config_path(cfg_lbl=cfg_label))

        logger.info(f"Processed Cf config {cfg_label}!")
