import argparse
import pathlib
from dataclasses import dataclass

from cfdmod.use_cases.block_gen import *
import pymeshlab


@dataclass
class ArgsModel:
    """Command line arguments for client app"""

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
    cfg = GenerationParams.from_file(pathlib.Path(args_use.config))
    vertices, triangles = build_single_block(cfg.block_params)

    single_line_repeat = (
        cfg.N_blocks_y - 1
        if cfg.spacing_params.offset_direction == "y"
        else cfg.N_blocks_x - 1
    )
    single_line_spacing = (
        cfg.spacing_params.spacing_x + cfg.block_params.length
        if cfg.spacing_params.offset_direction == "x"
        else cfg.spacing_params.spacing_y + cfg.block_params.width
    )
    multiple_line_repeat = (
        cfg.N_blocks_x - 1
        if cfg.spacing_params.offset_direction == "y"
        else cfg.N_blocks_y - 1
    )
    multiple_line_spacing = (
        cfg.spacing_params.spacing_x + cfg.block_params.length
        if cfg.spacing_params.offset_direction == "y"
        else cfg.spacing_params.spacing_y + cfg.block_params.width
    )

    single_line_vertices, single_line_triangles = linear_pattern(
        vertices,
        triangles,
        direction=cfg.spacing_params.offset_direction,
        n_repeats=single_line_repeat,
        spacing_value=single_line_spacing,
    )

    full_vertices, full_triangles = linear_pattern(
        single_line_vertices,
        single_line_triangles,
        direction=cfg.perpendicular_direction,
        n_repeats=multiple_line_repeat,
        spacing_value=multiple_line_spacing,
        offset_value=cfg.calculate_spacing(direction=cfg.perpendicular_direction),
    )

    m = pymeshlab.Mesh(full_vertices, full_triangles)
    ms = pymeshlab.MeshSet()
    ms.add_mesh(m, "cube_mesh")
    ms.save_current_mesh(
        str(pathlib.Path(args_use.config).parents[0] / "generated_cubes.stl")
    )
