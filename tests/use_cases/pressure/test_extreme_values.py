import unittest

import numpy as np

from cfdmod.use_cases.pressure.extreme_values import (
    ExtremeValuesParameters,
    calculate_extreme_values,
    fit_gumbel_model,
)


class TestExtremeValuesCalculation(unittest.TestCase):
    def setUp(self):
        self.timestep_arr = np.linspace(0, 10, 100)
        self.hist_series = np.linspace(0.1, 0.6, 100)
        self.params = ExtremeValuesParameters(CST_real=1, CST_sim=1, t=1, T0=1, T1=2, yR=1.4)

    def test_fit_gumbel_model(self):
        result = fit_gumbel_model(self.hist_series, self.params)
        self.assertEqual(round(result, ndigits=3), 0.527)

    def test_calculate_extreme_values(self):
        result = calculate_extreme_values(self.params, self.timestep_arr, self.hist_series)

        self.assertIsInstance(result, tuple)
        self.assertEqual(len(result), 2)
        self.assertIsInstance(result[0], float)
        self.assertIsInstance(result[1], float)


if __name__ == "__main__":
    unittest.main()
