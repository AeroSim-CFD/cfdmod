import pathlib
import unittest

import pandas as pd

from cfdmod.use_cases.pressure.cp_data import (
    calculate_statistics,
    read_pressure_data,
    transform_to_cp,
)


class TestCpData(unittest.TestCase):
    def test_read_and_slice_data(self):
        # Create dummy HDF5 files for testing
        static_data_path = pathlib.Path("./fixtures/tests/pressure/static_test.h5")
        body_data_path = pathlib.Path("./fixtures/tests/pressure/body_test.h5")
        pd.DataFrame({"time_step": [1, 2, 3, 4, 5], "rho": [1, 1, 1, 1, 1]}).to_hdf(
            static_data_path, "df"
        )
        pd.DataFrame({"time_step": [1, 2, 3, 4, 5], "rho": [1.1, 1.2, 1.3, 1.4, 1.5]}).to_hdf(
            body_data_path, "df"
        )

        press_data, body_data = read_pressure_data(static_data_path, body_data_path, (2, 4))

        self.assertEqual(press_data["time_step"].tolist(), [2, 3, 4])
        self.assertEqual(body_data["time_step"].tolist(), [2, 3, 4])

        static_data_path.unlink()
        body_data_path.unlink()

    def test_transform_instantaneous(self):
        press_data = pd.DataFrame({"time_step": [1, 2, 3, 4, 5], "rho": [1, 1, 1, 1, 1]})
        body_data = pd.DataFrame({"time_step": [1, 2, 3, 4, 5], "rho": [1.1, 1.2, 1.3, 1.4, 1.5]})

        transformed_data = transform_to_cp(press_data, body_data, 0.05, "instantaneous")
        self.assertIn("cp", transformed_data.columns)

    def test_transform_average(self):
        press_data = pd.DataFrame({"time_step": [1, 2, 3, 4, 5], "rho": [1, 1, 1, 1, 1]})
        body_data = pd.DataFrame({"time_step": [1, 2, 3, 4, 5], "rho": [1.1, 1.2, 1.3, 1.4, 1.5]})

        transformed_data = transform_to_cp(press_data, body_data, 0.05, "average")
        self.assertIn("cp", transformed_data.columns)

    def test_statistics_calculation(self):
        body_data = pd.DataFrame({"point_idx": [1, 1, 2, 2], "cp": [1.1, 1.2, 2.1, 2.2]})
        statistics_data = calculate_statistics(
            body_data, ["max", "min", "std", "avg", "skewness", "kurtosis"]
        )

        self.assertIn("cp_avg", statistics_data.columns)
        self.assertIn("cp_min", statistics_data.columns)
        self.assertIn("cp_max", statistics_data.columns)
        self.assertIn("cp_rms", statistics_data.columns)
        self.assertIn("cp_skewness", statistics_data.columns)
        self.assertIn("cp_kurtosis", statistics_data.columns)


if __name__ == "__main__":
    unittest.main()
