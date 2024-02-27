import pathlib
import unittest

from cfdmod.analysis.inflow.profile import InflowData, NormalizationParameters


class TestInflowProfile(unittest.TestCase):
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
            result = inflow_data.calculate_mean_velocity(for_components=["ux", "uy", "uz"])
            self.assertTrue(all([f"{c}_mean" in result.columns for c in ["ux", "uy", "uz"]]))

    def test_calculate_turbulence_intensity(self):
        for inflow_data in self.inflow_data.values():
            result = inflow_data.calculate_turbulence_intensity(for_components=["ux", "uy", "uz"])
            self.assertTrue(all([f"I_{c}" in result.columns for c in ["ux", "uy", "uz"]]))

    def test_calculate_spectral_density(self):
        normalization_params = NormalizationParameters(
            reference_velocity=1.0, characteristic_length=1.0
        )
        for inflow_data in self.inflow_data.values():
            result = inflow_data.calculate_spectral_density(
                target_index=0,
                for_components=["ux", "uy", "uz"],
                normalization_params=normalization_params,
            )
            self.assertTrue(all([f"S ({c})" in result.columns for c in ["ux", "uy", "uz"]]))
            self.assertTrue(all([f"f ({c})" in result.columns for c in ["ux", "uy", "uz"]]))

    def test_calculate_autocorrelation(self):
        for inflow_data in self.inflow_data.values():
            result = inflow_data.calculate_autocorrelation(
                anchor_point_idx=0, for_components=["ux", "uy", "uz"]
            )
            self.assertTrue(all([f"coef_{c}" in result.columns for c in ["ux", "uy", "uz"]]))


if __name__ == "__main__":
    unittest.main()
