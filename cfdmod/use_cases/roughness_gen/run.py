import pathlib

from cfdmod.api.geometry.STL import export_stl
from cfdmod.use_cases.roughness_gen.build_element import build_single_element
from cfdmod.use_cases.roughness_gen.linear_pattern import linear_pattern
from cfdmod.use_cases.roughness_gen.parameters import GenerationParams, RadialParams
from cfdmod.use_cases.roughness_gen.radial_pattern import radial_pattern


def run_linear(cfg: GenerationParams, output_path: pathlib.Path):
    """Orchestrate linear roughness element generation and write STL to output_path.

    Args:
        cfg (GenerationParams): Generation configuration for the linear pattern.
        output_path (pathlib.Path): Directory where roughness_elements.stl will be written.
    """
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


def run_radial(cfg: RadialParams, output_path: pathlib.Path):
    """Orchestrate radial roughness element generation and write STL to output_path.

    Args:
        cfg (RadialParams): Radial pattern configuration.
        output_path (pathlib.Path): Directory where roughness_elements.stl will be written.
    """
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
