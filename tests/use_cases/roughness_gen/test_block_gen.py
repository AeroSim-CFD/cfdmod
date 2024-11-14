import pathlib

import numpy as np
import pytest

from cfdmod.api.geometry.STL import export_stl
from cfdmod.use_cases.roughness_gen import (
    ElementParams,
    GenerationParams,
    SpacingParams,
    build_single_element,
    linear_pattern,
)


@pytest.mark.parametrize(
    "height,width,nx,ny",
    [
        (0.001, 0.001, 10, 10),
        (1, 1, 10, 10),
        (1, 1, 1000, 1000),
        (100, 100, 10, 10),
        (100, 100, 1000, 1000),
    ],
)
def test_element_generation(height, width, nx, ny):
    output_path = pathlib.Path("./output/roughness_gen")

    element_params = ElementParams(height=height, width=width)
    spacing_params = SpacingParams(
        spacing=(1, 1),
        line_offset=1,
        offset_direction="x",
    )
    cfg = GenerationParams(
        N_elements_x=nx,
        N_elements_y=ny,
        element_params=element_params,
        spacing_params=spacing_params,
    )

    triangles, normals = build_single_element(cfg.element_params)
    for n in normals:
        np.testing.assert_equal(n, [-1, 0, 0])

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

    export_stl(output_path / "roughness_full.stl", full_triangles, full_normals)
    export_stl(output_path / "roughness_line.stl", single_line_triangles, single_line_normals)
    export_stl(output_path / "roughness_element.stl", triangles, normals)

    assert len(triangles) == len(normals) == 2
    assert (
        len(single_line_triangles)
        == len(single_line_normals)
        == 2 * (cfg.single_line_elements + 1)
    )
    assert (
        len(full_triangles)
        == len(full_normals)
        == 2 * (cfg.single_line_elements + 1) * (cfg.multi_line_elements + 1)
    )
