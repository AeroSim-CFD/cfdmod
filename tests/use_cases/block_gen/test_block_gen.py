import pathlib
import unittest

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
        block_params = BlockParams(height=5, width=5, length=5)
        spacing_params = SpacingParams(
            spacing_x=2,
            spacing_y=2,
            line_offset=5,
            is_abs=False,
            offset_direction=OffsetDirection.x,
        )
        cfg = GenerationParams(
            N_blocks_x=10, N_blocks_y=10, block_params=block_params, spacing_params=spacing_params
        )

        vertices, triangles = build_single_block(cfg.block_params)

        single_line_vertices, single_line_triangles = linear_pattern(
            vertices,
            triangles,
            direction=cfg.spacing_params.offset_direction.value,
            n_repeats=cfg.single_line_blocks,
            spacing_value=cfg.single_line_spacing,
        )

        full_vertices, full_triangles = linear_pattern(
            single_line_vertices,
            single_line_triangles,
            direction=cfg.perpendicular_direction.value,
            n_repeats=cfg.multi_line_blocks,
            spacing_value=cfg.multi_line_spacing,
            offset_value=cfg.offset_spacing,
        )

        self.assertEqual(len(vertices), 8)
        self.assertEqual(len(triangles), 10)

        self.assertEqual(len(single_line_vertices), 8 * (cfg.single_line_blocks + 1))
        self.assertEqual(len(single_line_triangles), 10 * (cfg.single_line_blocks + 1))

        self.assertEqual(
            len(full_vertices), 8 * (cfg.single_line_blocks + 1) * (cfg.multi_line_blocks + 1)
        )
        self.assertEqual(
            len(full_triangles), 10 * (cfg.single_line_blocks + 1) * (cfg.multi_line_blocks + 1)
        )


if __name__ == "__main__":
    unittest.main()
