import unittest

import numpy as np
import pandas as pd
from lnas import LnasGeometry

from cfdmod.use_cases.pressure.extreme_values import ExtremeValuesParameters
from cfdmod.use_cases.pressure.statistics import Statistics
from cfdmod.use_cases.pressure.zoning.processing import (
    calculate_statistics,
    combine_stats_data_with_mesh,
    get_indexing_mask,
)
from cfdmod.use_cases.pressure.zoning.zoning_model import ZoningModel


class TestProcessingFunctions(unittest.TestCase):
    def setUp(self) -> None:
        self.stats_to_apply: list[Statistics] = [
            "max",
            "min",
            "std",
            "mean",
            "mean_qs",
            "skewness",
            "kurtosis",
            "xtr_min",
            "xtr_max",
        ]
        self.zoning = ZoningModel(x_intervals=[0, 5, 10], y_intervals=[0, 10], z_intervals=[0, 10])
        self.zoning.offset_limits(0.1)
        self.mesh = LnasGeometry(
            vertices=np.array([[0, 0, 0], [0, 10, 0], [10, 0, 0], [10, 10, 0]]),
            triangles=np.array([[0, 1, 2], [2, 1, 3]]),
        )
        self.variable = "cp"
        self.params = ExtremeValuesParameters(
            CST_real=1,
            CST_sim=1,
            t=2,
            T0=10,
            T1=100,
        )
        time_values = np.array([i / 2.0 for i in range(1, 201)])
        region_idx_values = np.array(
            [0] * int(len(time_values) / 2) + [3] * int(len(time_values) / 2)
        )
        cp_values = np.random.uniform(-1.0, 1.0, len(time_values))
        data = {
            "region_idx": region_idx_values,
            "cp": cp_values,
            "time_step": time_values,
        }
        self.hist_series = pd.DataFrame(data)
        return super().setUp()

    def test_calculate_statistics(self):
        stats = calculate_statistics(
            historical_data=self.hist_series,
            statistics_to_apply=self.stats_to_apply,
            variables=[self.variable],
            group_by_key="region_idx",
            extreme_params=self.params,
        )
        self.assertTrue(
            all([f"{self.variable}_{s}" in stats.columns for s in self.stats_to_apply])
        )

    def test_get_indexing_mask(self):
        df_regions = self.zoning.get_regions_df()
        region_mask = get_indexing_mask(self.mesh, df_regions)

        self.assertEqual(len(region_mask), len(self.mesh.triangles))
        self.assertEqual(region_mask[0], 0)
        self.assertEqual(region_mask[1], 1)

    def test_combine_stats_data_with_mesh(self):
        df_regions = self.zoning.get_regions_df()
        stats = calculate_statistics(
            historical_data=self.hist_series,
            statistics_to_apply=self.stats_to_apply,
            variables=[self.variable],
            group_by_key="region_idx",
            extreme_params=self.params,
        )
        idx_arr = get_indexing_mask(self.mesh, df_regions)
        result = combine_stats_data_with_mesh(self.mesh, idx_arr, stats)

        vars_match = [
            f"{self.variable}_{stats_var}" in result.columns for stats_var in self.stats_to_apply
        ]
        points_match = [i in result.point_idx for i in range(len(self.mesh.triangles))]

        self.assertTrue(all(vars_match))
        self.assertTrue(all(points_match))


if __name__ == "__main__":
    unittest.main()
