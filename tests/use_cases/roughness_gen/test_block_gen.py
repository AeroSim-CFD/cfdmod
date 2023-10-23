import pathlib
import unittest

from cfdmod.api.geometry.STL import export_stl
from cfdmod.use_cases.roughness_gen import (
    ElementParams,
    GenerationParams,
    SpacingParams,
    build_single_element,
    linear_pattern,
)


class TestElementGenerationUseCase(unittest.TestCase):
    def test_element_generation(self):
        output_path = pathlib.Path("./output/roughness_gen")

        element_params = ElementParams(height=1, width=1)
        spacing_params = SpacingParams(
            spacing=(1, 1),
            line_offset=1,
            offset_direction="x",
        )
        cfg = GenerationParams(
            N_elements_x=10,
            N_elements_y=10,
            element_params=element_params,
            spacing_params=spacing_params,
        )

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

        export_stl(output_path / "roughness_full.stl", full_triangles, full_normals)
        export_stl(output_path / "roughness_line.stl", single_line_triangles, single_line_normals)
        export_stl(output_path / "roughness_element.stl", triangles, normals)

        self.assertEqual(len(triangles), len(normals), 10)
        self.assertEqual(
            len(single_line_triangles),
            len(single_line_normals),
            10 * (cfg.single_line_elements + 1),
        )
        self.assertEqual(
            len(full_triangles),
            len(full_normals),
            10 * (cfg.single_line_elements + 1) * (cfg.multi_line_elements + 1),
        )


if __name__ == "__main__":
    unittest.main()
