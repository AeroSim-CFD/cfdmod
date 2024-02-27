import pathlib
import unittest

import numpy as np

from cfdmod.analysis.inflow.functions import (
    calculate_autocorrelation,
    calculate_mean_velocity,
    calculate_spectral_density,
    calculate_turbulence_intensity,
    spectral_density_function,
)
from cfdmod.analysis.inflow.profile import InflowData, NormalizationParameters


class TestInflowFunctions(unittest.TestCase):
    def setUp(self):
        self.inflow_data = {}
        folder_path = pathlib.Path("./fixtures/tests/inflow/pitot_inlet")
        self.inflow_data["old_version"] = InflowData.from_files(
            hist_series_path=folder_path / "hist_series.csv",
            points_path=folder_path / "points.csv",
        )
        folder_path = pathlib.Path("./fixtures/tests/inflow/autocorrelacao_10_h5")
        self.inflow_data["new_version"] = InflowData.from_files(
            hist_series_path=folder_path / "hist_series.h5",
            points_path=folder_path / "points.csv",
        )
        super().setUp()

    def test_calculate_mean_velocity(self):
        for inflow_data in self.inflow_data.values():
            result = calculate_mean_velocity(
                inflow_data=inflow_data, for_components=["ux", "uy", "uz"]
            )
            self.assertTrue(all([f"{c}_mean" in result.columns for c in ["ux", "uy", "uz"]]))

    def test_calculate_turbulence_intensity(self):
        for inflow_data in self.inflow_data.values():
            result = calculate_turbulence_intensity(
                inflow_data=inflow_data, for_components=["ux", "uy", "uz"]
            )
            self.assertTrue(all([f"I_{c}" in result.columns for c in ["ux", "uy", "uz"]]))

    def test_calculate_spectral_density(self):
        normalization_params = NormalizationParameters(
            reference_velocity=1.0, characteristic_length=1.0
        )
        for inflow_data in self.inflow_data.values():
            result = calculate_spectral_density(
                inflow_data=inflow_data,
                target_index=0,
                for_components=["ux", "uy", "uz"],
                normalization_params=normalization_params,
            )
            self.assertTrue(all([f"S ({c})" in result.columns for c in ["ux", "uy", "uz"]]))
            self.assertTrue(all([f"f ({c})" in result.columns for c in ["ux", "uy", "uz"]]))

    def test_calculate_autocorrelation(self):
        for inflow_data in self.inflow_data.values():
            result = calculate_autocorrelation(
                inflow_data=inflow_data, anchor_point_idx=0, for_components=["ux", "uy", "uz"]
            )
            self.assertTrue(all([f"coef_{c}" in result.columns for c in ["ux", "uy", "uz"]]))

    def test_spectral_density(self):
        timestamps = np.linspace(0, 10, 1000)
        velocity_signal = np.sin(2 * np.pi * 1 * timestamps) + 0.5 * np.random.randn(1000)
        xf, yf = spectral_density_function(velocity_signal, timestamps, 1.0, 1.0)

        self.assertEqual(len(xf), len(yf), "Lengths of output arrays do not match")
        self.assertIsInstance(xf, np.ndarray, "xf is not a numpy array")
        self.assertIsInstance(yf, np.ndarray, "yf is not a numpy array")


if __name__ == "__main__":
    unittest.main()
