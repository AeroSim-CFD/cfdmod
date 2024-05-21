import unittest
from typing import get_args

import numpy as np
import pandas as pd
from lnas import LnasGeometry

from cfdmod.use_cases.pressure.statistics import Statistics
from cfdmod.use_cases.pressure.zoning.processing import (
    combine_stats_data_with_mesh,
    get_indexing_mask,
)
from cfdmod.use_cases.pressure.zoning.zoning_model import ZoningModel


class TestProcessingFunctions(unittest.TestCase):
    def setUp(self) -> None:
        self.zoning = ZoningModel(x_intervals=[0, 5, 10], y_intervals=[0, 10], z_intervals=[0, 10])
        self.zoning.offset_limits(0.1)
        self.mesh = LnasGeometry(
            vertices=np.array([[0, 0, 0], [0, 10, 0], [10, 0, 0], [10, 10, 0]]),
            triangles=np.array([[0, 1, 2], [2, 1, 3]]),
        )
        region_idx_values = np.array([0, 3], dtype=np.int32)
        data = {
            "region_idx": region_idx_values,
            "mean": [0.5, 0.7],
            "rms": [0.1, 0.15],
            "max": [1.2, 0.08],
            "min": [-0.9, -1.5],
        }
        self.stats = pd.DataFrame(data)
        return super().setUp()

    def test_get_indexing_mask(self):
        df_regions = self.zoning.get_regions_df()
        region_mask = get_indexing_mask(self.mesh, df_regions)

        self.assertEqual(len(region_mask), len(self.mesh.triangles))
        self.assertEqual(region_mask[0], 0)
        self.assertEqual(region_mask[1], 1)

    def test_combine_stats_data_with_mesh(self):
        df_regions = self.zoning.get_regions_df()
        idx_arr = get_indexing_mask(self.mesh, df_regions)
        result = combine_stats_data_with_mesh(self.mesh, idx_arr, self.stats)

        vars_match = ["mean", "rms", "min", "max"]
        points_match = [i in result.point_idx for i in range(len(self.mesh.triangles))]

        self.assertTrue(all(vars_match))
        self.assertTrue(all(points_match))


if __name__ == "__main__":
    unittest.main()
