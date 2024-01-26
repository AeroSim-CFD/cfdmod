import argparse
import pathlib
from dataclasses import dataclass

from lnas import LnasFormat

from cfdmod.logger import logger
from cfdmod.use_cases.pressure.force.Cf_config import CfCaseConfig
from cfdmod.use_cases.pressure.force.Cf_data import process_Cf
from cfdmod.use_cases.pressure.geometry import get_excluded_entities
from cfdmod.use_cases.pressure.output import CommonOutput
from cfdmod.use_cases.pressure.path_manager import CfPathManager


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
    path_manager = CfPathManager(output_path=pathlib.Path(args_use.output))

    cfg_path = pathlib.Path(args_use.config)
    mesh_path = pathlib.Path(args_use.mesh)
    cp_path = pathlib.Path(args_use.cp)

    post_proc_cfg = CfCaseConfig.from_file(cfg_path)

    logger.info("Reading mesh description...")
    mesh = LnasFormat.from_file(mesh_path)
    logger.info("Mesh description loaded successfully!")

    for cfg_label, cfg in post_proc_cfg.force_coefficient.items():
        full_processed_entities = []
        included_sfcs = []
        for body_lbl in cfg.bodies:
            logger.info(f"Processing body {body_lbl} ...")

            cf_output: CommonOutput = process_Cf(
                mesh=mesh,
                body_cfg=post_proc_cfg.bodies[body_lbl],
                cfg=cfg,
                cp_path=cp_path,
                extreme_params=post_proc_cfg.extreme_values,
            )

            full_processed_entities += cf_output.processed_entities
            included_sfcs += post_proc_cfg.bodies[body_lbl].surfaces
            logger.info(f"Processed body {body_lbl}!")

        excluded_sfcs = [k for k in mesh.surfaces.keys() if k not in included_sfcs]
        col = full_processed_entities[0].stats_df.columns
        excluded_entity = get_excluded_entities(
            excluded_sfc_list=excluded_sfcs, mesh=mesh, data_columns=col
        )
        full_processed_entities += [excluded_entity]

        # cf_output.save_outputs(
        #     file_lbl=body_lbl, cfg_label=cfg_label, path_manager=path_manager
        # )
