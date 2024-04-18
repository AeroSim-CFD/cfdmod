import unittest

import numpy as np

from cfdmod.use_cases.pressure.extreme_values import (
    ExtremeValuesParameters,
    fit_gumbel_model,
    gumbel_extreme_values,
    moving_average_extreme_values,
)


class TestExtremeValuesCalculation(unittest.TestCase):
    def setUp(self):
        self.timestep_arr = np.linspace(0, 10, 100)
        self.hist_series = np.linspace(0.1, 0.6, 100)
        self.gumbell_params = ExtremeValuesParameters(
            CST_real=1,
            CST_sim=1,
            extreme_model="Gumbell",
            parameters={"t": 1, "T0": 1, "T1": 2, "yR": 1.4},
        )
        self.moving_avg_params = ExtremeValuesParameters(
            CST_real=1,
            CST_sim=1,
            extreme_model="Moving average",
            parameters={"window_size_real": 2},
        )

    def test_fit_gumbel_model(self):
        result = fit_gumbel_model(self.hist_series, self.gumbell_params)
        self.assertEqual(round(result, ndigits=3), 0.527)

    def test_calculate_gumbel_extreme_values(self):
        result = gumbel_extreme_values(self.gumbell_params, self.timestep_arr, self.hist_series)

        self.assertIsInstance(result, tuple)
        self.assertEqual(len(result), 2)
        self.assertIsInstance(result[0], float)
        self.assertIsInstance(result[1], float)

    def test_calculate_moving_avg_extreme_values(self):
        result = moving_average_extreme_values(self.moving_avg_params, self.hist_series)

        self.assertIsInstance(result, tuple)
        self.assertEqual(len(result), 2)
        self.assertIsInstance(result[0], float)
        self.assertIsInstance(result[1], float)


if __name__ == "__main__":
    unittest.main()
