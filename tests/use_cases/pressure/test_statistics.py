import unittest

import numpy as np
import pandas as pd

from cfdmod.use_cases.pressure.statistics import (
    BasicStatisticModel,
    ExtremeAbsoluteParamsModel,
    ExtremeGumbelParamsModel,
    ExtremeMovingAverageParamsModel,
    ExtremePeakParamsModel,
    MeanEquivalentParamsModel,
    ParameterizedStatisticModel,
)
from cfdmod.use_cases.pressure.zoning.processing import calculate_statistics
from cfdmod.utils import convert_dataframe_into_matrix


class TestStatistics(unittest.TestCase):
    def setUp(self) -> None:
        self.stats_to_apply: list[ParameterizedStatisticModel | BasicStatisticModel] = [
            BasicStatisticModel(stats="mean"),
            BasicStatisticModel(stats="rms"),
            BasicStatisticModel(stats="skewness"),
            BasicStatisticModel(stats="kurtosis"),
            ParameterizedStatisticModel(
                stats="mean_eq", params=MeanEquivalentParamsModel(scale_factor=0.61)
            ),
            ParameterizedStatisticModel(stats="min", params=ExtremeAbsoluteParamsModel()),
            ParameterizedStatisticModel(stats="min", params=ExtremePeakParamsModel(peak_factor=3)),
            ParameterizedStatisticModel(
                stats="max",
                params=ExtremeMovingAverageParamsModel(
                    window_size_interval=3,
                    full_scale_characteristic_length=22.4,
                    full_scale_U_H=40,
                ),
            ),
            ParameterizedStatisticModel(
                stats="max",
                params=ExtremeGumbelParamsModel(
                    peak_duration=3,
                    event_duration=60,
                    n_subdivisions=10,
                    non_exceedance_probability=0.78,
                    full_scale_characteristic_length=22.4,
                    full_scale_U_H=40,
                ),
            ),
        ]
        time_values = np.array([i for i in range(0, 500)], dtype=np.float32)
        idx_values = np.array([i for i in range(0, 200)], dtype=np.int32)
        values = np.random.uniform(-1.0, 1.0, len(time_values) * len(idx_values))
        data = {
            "point_idx": np.tile(idx_values, len(time_values)),
            "cp": values,
            "time_normalized": np.repeat(time_values, len(idx_values)),
        }
        self.hist_series = convert_dataframe_into_matrix(
            pd.DataFrame(data), row_data_label="time_normalized", value_data_label="cp"
        )
        return super().setUp()

    def test_calculate_statistics(self):
        stats = calculate_statistics(
            historical_data=self.hist_series, statistics_to_apply=self.stats_to_apply
        )
        self.assertTrue(all([s.stats in stats.columns for s in self.stats_to_apply]))
        self.assertFalse(stats.isnull().values.any())
        self.assertFalse(stats.empty)


if __name__ == "__main__":
    unittest.main()
