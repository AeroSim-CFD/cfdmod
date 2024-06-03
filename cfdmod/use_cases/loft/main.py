import argparse
import pathlib
from dataclasses import dataclass

import numpy as np

from cfdmod.api.geometry.STL import export_stl, read_stl
from cfdmod.logger import logger
from cfdmod.use_cases.loft.functions import (
    apply_remeshing,
    generate_loft_surface,
    rotate_vector_around_z,
)
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
    triangles, normals = read_stl(mesh_path)

    for case_lbl, loft_params in cfg.cases.items():
        for wind_angle in loft_params.wind_source_angles:
            logger.info(f"Generating loft for {case_lbl}/{wind_angle}...")
            wind_source_direction = rotate_vector_around_z(
                np.array(cfg.reference_direction, dtype=np.float32), wind_angle
            )
            loft_directions = {
                "upwind": -np.array(wind_source_direction),
                "downwind": np.array(wind_source_direction),
            }

            for side, direction in loft_directions.items():
                loft_tri, loft_normals = generate_loft_surface(
                    triangle_vertices=triangles,
                    projection_diretion=direction,
                    loft_length=loft_params.loft_length,
                    loft_z_pos=loft_params.upwind_elevation,
                    filter_radius=loft_params.filter_radius,
                )

                export_stl(
                    output_path / f"{case_lbl}" / f"{side}_loft.stl",
                    loft_tri,
                    loft_normals,
                )
                apply_remeshing(
                    element_size=loft_params.mesh_element_size,
                    mesh_path=output_path / f"{case_lbl}" / f"{side}_loft.stl",
                    output_path=output_path / f"{case_lbl}" / f"{side}_loft.stl",
                )
            logger.info(f"Generated loft for {case_lbl}/{wind_angle}!")
