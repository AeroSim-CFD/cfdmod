import pathlib
import unittest

from cfdmod.api.geometry.STL import export_stl
from cfdmod.use_cases.block_gen import (
    BlockParams,
    GenerationParams,
    OffsetDirection,
    SpacingParams,
    build_single_block,
    linear_pattern,
)


class TestBlockGenerationUseCase(unittest.TestCase):
    def test_block_generation(self):
        output_path = pathlib.Path("./output/block_gen")

        block_params = BlockParams(height=1, width=1, length=1)
        spacing_params = SpacingParams(
            spacing_x=1,
            spacing_y=1,
            line_offset=1,
            is_abs=True,
            offset_direction=OffsetDirection.x,
        )
        cfg = GenerationParams(
            N_blocks_x=10,
            N_blocks_y=10,
            block_params=block_params,
            spacing_params=spacing_params,
        )

        triangles, normals = build_single_block(cfg.block_params)

        single_line_triangles, single_line_normals = linear_pattern(
            triangles,
            normals,
            direction=cfg.spacing_params.offset_direction.value,
            n_repeats=cfg.single_line_blocks,
            spacing_value=cfg.single_line_spacing,
        )

        full_triangles, full_normals = linear_pattern(
            single_line_triangles,
            single_line_normals,
            direction=cfg.perpendicular_direction.value,
            n_repeats=cfg.multi_line_blocks,
            spacing_value=cfg.multi_line_spacing,
            offset_value=cfg.offset_spacing,
        )

        export_stl(output_path / "blocks_full.stl", full_triangles, full_normals)
        export_stl(output_path / "blocks_line.stl", single_line_triangles, single_line_normals)
        export_stl(output_path / "block.stl", triangles, normals)

        self.assertEqual(len(triangles), len(normals), 10)
        self.assertEqual(
            len(single_line_triangles), len(single_line_normals), 10 * (cfg.single_line_blocks + 1)
        )
        self.assertEqual(
            len(full_triangles),
            len(full_normals),
            10 * (cfg.single_line_blocks + 1) * (cfg.multi_line_blocks + 1),
        )


if __name__ == "__main__":
    unittest.main()
