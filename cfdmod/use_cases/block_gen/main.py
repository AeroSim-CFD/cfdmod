import argparse
import pathlib
from dataclasses import dataclass

from cfdmod.api.geometry.STL import export_stl
from cfdmod.use_cases.block_gen import *


@dataclass
class ArgsModel:
    """Command line arguments for client app"""

    config: str
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
        help="Path to config .yaml file",
        type=str,
    )
    ap.add_argument(
        "--output",
        required=True,
        help="Output path for stl file",
        type=str,
    )
    parsed_args = ap.parse_args(args)
    args_model = ArgsModel(**vars(parsed_args))
    return args_model


def main(*args):
    args_use = get_args_process(*args)
    cfg = GenerationParams.from_file(pathlib.Path(args_use.config))
    output_path = pathlib.Path(args_use.output)

    vertices, triangles = build_single_block(cfg.block_params)

    single_line_vertices, single_line_triangles = linear_pattern(
        vertices,
        triangles,
        direction=cfg.spacing_params.offset_direction,
        n_repeats=cfg.single_line_blocks,
        spacing_value=cfg.single_line_spacing,
    )

    full_vertices, full_triangles = linear_pattern(
        single_line_vertices,
        single_line_triangles,
        direction=cfg.perpendicular_direction,
        n_repeats=cfg.multi_line_blocks,
        spacing_value=cfg.multi_line_spacing,
        offset_value=cfg.calculate_spacing(direction=cfg.perpendicular_direction),
    )

    export_stl(output_path / "block_gen.stl", full_vertices, full_triangles)
