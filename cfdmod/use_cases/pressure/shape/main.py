import argparse
import pathlib
from dataclasses import dataclass

import pandas as pd
from lnas import LnasFormat

from cfdmod.api.vtk.write_vtk import merge_polydata, vtkPolyData, write_polydata
from cfdmod.logger import logger
from cfdmod.use_cases.pressure.geometry import create_NaN_polydata, get_excluded_surfaces
from cfdmod.use_cases.pressure.path_manager import CePathManager
from cfdmod.use_cases.pressure.shape.Ce_config import CeCaseConfig
from cfdmod.use_cases.pressure.shape.Ce_data import get_surfaces_raw_data, process_surface


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

    post_proc_cfg = CeCaseConfig.from_file(cfg_path)

    logger.info("Reading mesh description...")
    mesh = LnasFormat.from_file(mesh_path)
    logger.info("Mesh description loaded successfully!")

    logger.info("Preparing to read pressure coefficients data...")
    cp_data = pd.read_hdf(cp_path)

    cp_data_to_use = cp_data.to_frame() if isinstance(cp_data, pd.Series) else cp_data
    logger.info("Read pressure coefficient data successfully!")

    n_timesteps = cp_data_to_use["time_step"].unique().shape[0]

    for cfg_label, cfg in post_proc_cfg.shape_coefficient.items():
        processed_polydata: list[vtkPolyData] = []
        logger.info(f"Processing {cfg_label} ...")

        sfc_dict = {set_lbl: sfc_list for set_lbl, sfc_list in cfg.sets.items()}
        sfc_dict |= {sfc: [sfc] for sfc in mesh.surfaces.keys() if sfc not in cfg.surfaces_in_sets}

        data_columns = []
        surfaces_to_process = get_surfaces_raw_data(surface_dict=sfc_dict, cfg=cfg, mesh=mesh)
        for sfc_lbl, raw_surface in surfaces_to_process.items():
            logger.info(f"Processing surface {sfc_lbl}")

            processed_surface = process_surface(
                raw_surface=raw_surface,
                cfg=cfg,
                cp_data=cp_data_to_use,
                n_timesteps=n_timesteps,
            )
            processed_surface.save_outputs(
                sfc_label=sfc_lbl, cfg_label=cfg_label, path_manager=path_manager
            )

            processed_polydata.append(processed_surface.polydata)
            data_columns = processed_surface.surface_ce_stats.columns

            logger.info(f"Processed surface {sfc_lbl}")

        sfc_list = [sfc for sfc in cfg.zoning.exclude if sfc in mesh.surfaces.keys()]  # type: ignore
        sfc_list += [
            sfc
            for set_lbl, sfc_set in cfg.sets.items()
            for sfc in sfc_set
            if set_lbl in cfg.zoning.exclude  # type: ignore
        ]
        if len(sfc_list) != 0:
            excluded_sfcs = get_excluded_surfaces(mesh=mesh, sfc_list=sfc_list)
            excluded_sfcs.export_stl(path_manager.get_excluded_surface_path(cfg_label))
            # Include polydata with NaN values
            columns = [col for col in data_columns if col not in ["point_idx", "region_idx"]]
            excluded_polydata = create_NaN_polydata(mesh=excluded_sfcs, column_labels=columns)
            processed_polydata.append(excluded_polydata)

        merged_polydata = merge_polydata(processed_polydata)
        write_polydata(path_manager.get_vtp_path(mesh.name, cfg_label), merged_polydata)

        logger.info(f"Merged and saved polydata for {cfg_label}")
