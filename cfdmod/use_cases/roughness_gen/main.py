import argparse
import pathlib
from dataclasses import dataclass

from cfdmod.api.geometry.STL import export_stl
from cfdmod.use_cases.roughness_gen import (
    GenerationParams,
    RadialParams,
    build_single_element,
    linear_pattern,
    radial_pattern,
)


@dataclass
class ArgsModel:
    """Command line arguments for client app"""

    config: str
    output: str
    mode: str


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
    ap.add_argument(
        "--mode",
        default="linear",
        choices=["linear", "radial"],
        help="Generation mode: linear (default) or radial",
        type=str,
    )
    parsed_args = ap.parse_args(args)
    args_model = ArgsModel(**vars(parsed_args))
    return args_model


def main(*args):
    args_use = get_args_process(*args)
    output_path = pathlib.Path(args_use.output)

    if args_use.mode == "radial":
        cfg = RadialParams.from_file(pathlib.Path(args_use.config))
        surface_paths = [pathlib.Path(p) for p in cfg.surfaces.values()]
        full_triangles, full_normals = radial_pattern(
            element_params=cfg.element_params,
            r_start=cfg.r_start,
            r_end=cfg.r_end,
            radial_spacing=cfg.radial_spacing,
            arc_spacing=cfg.arc_spacing,
            ring_offset_distance=cfg.ring_offset_distance,
            center=cfg.center,
            surface_paths=surface_paths,
        )
        export_stl(output_path / "roughness_elements.stl", full_triangles, full_normals)
        return

    cfg = GenerationParams.from_file(pathlib.Path(args_use.config))

    triangles, normals = build_single_element(cfg.element_params)

    single_line_triangles, single_line_normals = linear_pattern(
        triangles,
        normals,
        direction=cfg.spacing_params.offset_direction,
        n_repeats=cfg.single_line_elements,
        spacing_value=cfg.single_line_spacing,
    )

    full_triangles, full_normals = linear_pattern(
        single_line_triangles,
        single_line_normals,
        direction=cfg.perpendicular_direction,
        n_repeats=cfg.multi_line_elements,
        spacing_value=cfg.multi_line_spacing,
        offset_value=cfg.spacing_params.line_offset,
    )

    export_stl(output_path / "roughness_elements.stl", full_triangles, full_normals)
