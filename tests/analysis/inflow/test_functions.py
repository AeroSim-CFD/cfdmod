import unittest

import numpy as np

from cfdmod.analysis.inflow.functions import spectral_density


class TestInflowFunctions(unittest.TestCase):
    def test_spectral_density(self):
        timestamps = np.linspace(0, 10, 1000)
        velocity_signal = np.sin(2 * np.pi * 1 * timestamps) + 0.5 * np.random.randn(1000)
        xf, yf = spectral_density(velocity_signal, timestamps, 1.0, 1.0)

        self.assertEqual(len(xf), len(yf), "Lengths of output arrays do not match")
        self.assertIsInstance(xf, np.ndarray, "xf is not a numpy array")
        self.assertIsInstance(yf, np.ndarray, "yf is not a numpy array")


if __name__ == "__main__":
    unittest.main()
