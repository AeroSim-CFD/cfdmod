import argparse
import pathlib
from dataclasses import dataclass

import pandas as pd
from nassu.lnas import LagrangianFormat

from cfdmod.api.vtk.write_vtk import merge_polydata, write_polydata
from cfdmod.logger import logger
from cfdmod.use_cases.pressure.path_manager import CePathManager
from cfdmod.use_cases.pressure.shape.Ce_config import CeConfig
from cfdmod.use_cases.pressure.shape.Ce_data import process_surface


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
    path_manager = CePathManager(output_path=pathlib.Path(args_use.output))

    cfg_path = pathlib.Path(args_use.config)
    mesh_path = pathlib.Path(args_use.mesh)
    cp_path = pathlib.Path(args_use.cp)

    post_proc_cfg = CeConfig.from_file(cfg_path)

    logger.info("Reading mesh description...")
    mesh = LagrangianFormat.from_file(mesh_path)
    logger.info("Mesh description loaded successfully!")

    logger.info("Preparing to read pressure coefficients data...")
    cp_data = pd.read_hdf(cp_path)

    cp_data_to_use = cp_data.to_frame() if isinstance(cp_data, pd.Series) else cp_data
    logger.info("Read pressure coefficient data successfully!")

    n_timesteps = cp_data_to_use["time_step"].unique().shape[0]

    for cfg_label, cfg in post_proc_cfg.items():
        processed_polydata = []

        logger.info(f"Processing {cfg_label} ...")
        for sfc in mesh.surfaces.keys():
            if sfc in cfg.zoning.exclude:
                logger.info(f"Surface {sfc} ignored!")  # Ignore surface
                continue

            logger.info(f"Processing surface {sfc} ...")

            processed_surface = process_surface(
                body_mesh=mesh,
                sfc_label=sfc,
                cfg=cfg,
                cp_data=cp_data_to_use,
                n_timesteps=n_timesteps,
            )
            processed_surface.save_outputs(
                sfc_label=sfc, cfg_label=cfg_label, path_manager=path_manager
            )

            processed_polydata.append(processed_surface.polydata)

            logger.info(f"Processed surface {sfc}")

        # Output 5: VTK
        merged_polydata = merge_polydata(processed_polydata)
        write_polydata(path_manager.get_vtp_path(mesh.name, cfg_label), merged_polydata)

        logger.info(f"Merged polydata for {cfg_label}")
