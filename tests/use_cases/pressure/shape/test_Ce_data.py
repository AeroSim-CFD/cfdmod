import unittest

import numpy as np
import pandas as pd
from lnas import LnasGeometry

from cfdmod.use_cases.pressure.shape.Ce_data import calculate_statistics, transform_to_Ce
from cfdmod.use_cases.pressure.statistics import Statistics


class TestTransformToCeAndCalculateStatistics(unittest.TestCase):
    def setUp(self):
        vertices = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0]])
        triangles = np.array([[0, 1, 2]])
        self.surface_mesh = LnasGeometry(vertices, triangles)
        self.cp_data = pd.DataFrame(
            {"point_idx": [0, 0, 0], "cp": [0.1, 0.2, 0.3], "time_step": [0, 1, 2]}
        )
        self.region_data = pd.DataFrame(
            {
                "region_idx": [0, 0, 0, 0, 1, 1, 1, 1],
                "Ce": [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8],
            }
        )
        self.statistics_to_apply: list[Statistics] = [
            "mean",
            "min",
            "max",
            "std",
            "skewness",
            "kurtosis",
        ]

    def test_transform_to_Ce(self):
        sfc_triangles_idxs = np.array([0])
        triangles_region = np.array([0])
        n_timesteps = 3

        result = transform_to_Ce(
            self.surface_mesh, self.cp_data, sfc_triangles_idxs, triangles_region, n_timesteps
        )

        self.assertEqual(len(result), n_timesteps)  # Three timesteps x 1 triangle
        self.assertTrue("Ce" in result.columns)

    def test_calculate_statistics(self):
        # Test the function to calculate statistics
        result = calculate_statistics(self.region_data, self.statistics_to_apply, variables=["Ce"])

        self.assertEqual(len(result), 2)  # Two regions (0, 1)
        self.assertTrue((result.isnull().sum() == 0).all())


if __name__ == "__main__":
    unittest.main()
