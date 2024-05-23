import unittest

import numpy as np

from cfdmod.use_cases.pressure.extreme_values import (
    fit_gumbel_model,
    gumbel_extreme_values,
    moving_average_extreme_values,
    peak_extreme_values,
)
from cfdmod.use_cases.pressure.statistics import (
    ExtremeGumbelParamsModel,
    ExtremeMovingAverageParamsModel,
    ExtremePeakParamsModel,
)


class TestExtremeValuesCalculation(unittest.TestCase):
    def setUp(self):
        self.timestep_arr = np.linspace(0, 10, 100)
        self.hist_series = np.linspace(0.1, 0.6, 100)
        self.gumbell_params = ExtremeGumbelParamsModel(
            peak_duration=3,
            event_duration=60,
            non_exceedance_probability=0.78,
            full_scale_U_H=40,
            full_scale_characteristic_length=22.4,
        )
        self.moving_avg_params = ExtremeMovingAverageParamsModel(
            window_size_interval=2, full_scale_U_H=40, full_scale_characteristic_length=22.4
        )
        self.peak_params = ExtremePeakParamsModel(peak_factor=2)

    def test_fit_gumbel_model(self):
        result = fit_gumbel_model(self.hist_series, self.gumbell_params)
        self.assertEqual(round(result, ndigits=3), 0.712)

    def test_calculate_gumbel_extreme_values(self):
        result = gumbel_extreme_values(self.gumbell_params, self.timestep_arr, self.hist_series)

        self.assertEqual(len(result), 2)
        self.assertEqual(round(result[0], ndigits=3), 0.48)
        self.assertEqual(round(result[1], ndigits=3), 0.973)

    def test_calculate_moving_avg_extreme_values(self):
        result = moving_average_extreme_values(self.moving_avg_params, self.hist_series)

        self.assertEqual(len(result), 2)
        self.assertEqual(round(result[0], ndigits=3), 0.108)
        self.assertEqual(round(result[1], ndigits=3), 0.592)

    def test_calculate_peak_extreme_values(self):
        result = peak_extreme_values(self.peak_params, self.hist_series)

        self.assertEqual(len(result), 2)
        self.assertEqual(round(result[0], ndigits=3), 0.058)
        self.assertEqual(round(result[1], ndigits=3), 0.642)


if __name__ == "__main__":
    unittest.main()
