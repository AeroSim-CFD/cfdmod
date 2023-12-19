import unittest

import numpy as np

from cfdmod.use_cases.pressure.extreme_values import (
    ExtremeValuesParameters,
    calculate_extreme_values,
)


class TestExtremeValuesCalculation(unittest.TestCase):
    def test_calculate_extreme_values(self):
        params = ExtremeValuesParameters(CST_real=1, CST_sim=1, t=1, T0=1, T1=2, yR=1.4)
        timestep_arr = np.linspace(0, 10, 100)
        hist_series = np.random.rand(100)

        result = calculate_extreme_values(params, timestep_arr, hist_series)

        self.assertIsInstance(result, tuple)
        self.assertEqual(len(result), 2)
        self.assertIsInstance(result[0], float)
        self.assertIsInstance(result[1], float)


if __name__ == "__main__":
    unittest.main()
