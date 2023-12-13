import unittest

import numpy as np
import pandas as pd

from cfdmod.use_cases.pressure.extreme_values import ExtremeValuesParameters
from cfdmod.use_cases.pressure.statistics import Statistics
from cfdmod.use_cases.pressure.zoning.processing import calculate_statistics


class TestProcessingFunctions(unittest.TestCase):
    def setUp(self) -> None:
        self.stats_to_apply: list[Statistics] = ["mean_qs", "mean", "xtr_max", "xtr_min"]
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
            [1] * int(len(time_values) / 2) + [2] * int(len(time_values) / 2)
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


if __name__ == "__main__":
    unittest.main()
