import pathlib
import unittest

from cfdmod.analysis.inflow.profile import InflowData, NormalizationParameters


class TestInflowProfile(unittest.TestCase):
    def setUp(self):
        folder_path = pathlib.Path("./fixtures/tests/inflow/pitot_inlet")
        self.inflow_data = InflowData.from_folder(folder_path)
        super().setUp()

    def test_calculate_mean_velocity(self):
        result = self.inflow_data.calculate_mean_velocity(for_components=["ux", "uy", "uz"])
        self.assertTrue(all([f"{c}_mean" in result.columns for c in ["ux", "uy", "uz"]]))

    def test_calculate_turbulence_intensity(self):
        result = self.inflow_data.calculate_turbulence_intensity(for_components=["ux", "uy", "uz"])
        self.assertTrue(all([f"I_{c}" in result.columns for c in ["ux", "uy", "uz"]]))

    def test_calculate_spectral_density(self):
        normalization_params = NormalizationParameters(
            reference_velocity=1.0, characteristic_length=1.0
        )
        result = self.inflow_data.calculate_spectral_density(
            target_index=0,
            for_components=["ux", "uy", "uz"],
            normalization_params=normalization_params,
        )
        self.assertTrue(all([f"S ({c})" in result.columns for c in ["ux", "uy", "uz"]]))
        self.assertTrue(all([f"f ({c})" in result.columns for c in ["ux", "uy", "uz"]]))

    def test_calculate_autocorrelation(self):
        result = self.inflow_data.calculate_autocorrelation(
            anchor_point_idx=0, for_components=["ux", "uy", "uz"]
        )
        self.assertTrue(all([f"coef_{c}" in result.columns for c in ["ux", "uy", "uz"]]))


if __name__ == "__main__":
    unittest.main()
