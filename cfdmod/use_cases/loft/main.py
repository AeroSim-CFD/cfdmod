import argparse
import pathlib
from dataclasses import dataclass

import lnas
import numpy as np

from cfdmod.api.geometry.STL import export_stl, read_stl
from cfdmod.logger import logger
from cfdmod.use_cases.loft.functions import generate_loft_surface
from cfdmod.use_cases.loft.parameters import LoftCaseConfig


@dataclass
class ArgsModel:
    """Command line arguments for client app"""

    config: str
    surface: str
    output: str


def get_args_process(args: list[str]) -> ArgsModel:
    """Get arguments model from list of command line args

    Args:
        args (List[str]): List of command line arguments passed

    Returns:
        ArgsModel: Arguments model for client app
    """
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--config",
        required=True,
        help="Path to loft config file",
        type=str,
    )
    ap.add_argument(
        "--surface",
        required=True,
        help="Path to stl surface file",
        type=str,
    )
    ap.add_argument(
        "--output",
        required=True,
        help="Output path",
        type=str,
    )
    parsed_args = ap.parse_args(args)
    args_model = ArgsModel(**vars(parsed_args))
    return args_model


def main(*args):
    args_use = get_args_process(*args)
    cfg_file = pathlib.Path(args_use.config)
    mesh_path = pathlib.Path(args_use.surface)
    output_path = pathlib.Path(args_use.output)

    cfg = LoftCaseConfig.from_file(cfg_file)
    geom = lnas.LnasFormat.from_file(mesh_path).geometry

    for case_lbl, loft_params in cfg.cases.items():
        if case_lbl == "default":
            continue
        logger.info(f"Generating loft for {case_lbl}...")
        loft_geom = generate_loft_surface(
            geom=geom,
            loft_radius=loft_params.loft_radius,
            loft_z_pos=loft_params.upwind_elevation,
        )
        loft_tris = loft_geom.triangle_vertices
        u = loft_tris[:, 1] - loft_tris[:, 0]
        v = loft_tris[:, 2] - loft_tris[:, 0]
        loft_normals = np.cross(u, v)
        export_stl(
            output_path / f"{case_lbl}" / "loft.stl",
            loft_tris,
            loft_normals,
        )
        logger.info(f"Generated loft for {case_lbl}!")
