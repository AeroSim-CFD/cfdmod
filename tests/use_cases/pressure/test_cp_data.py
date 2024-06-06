import unittest

import pandas as pd

from cfdmod.use_cases.pressure.cp_data import filter_data, transform_to_cp


class TestCpData(unittest.TestCase):
    def test_read_and_slice_data(self):
        press_data = pd.DataFrame({"time_step": [1, 2, 3, 4, 5], "0": [1, 1, 1, 1, 1]})
        body_data = pd.DataFrame({"time_step": [1, 2, 3, 4, 5], "0": [1.1, 1.2, 1.3, 1.4, 1.5]})

        press_data = filter_data(press_data, (2, 4))
        body_data = filter_data(body_data, (2, 4))

        self.assertEqual(press_data["time_step"].tolist(), [2, 3, 4])
        self.assertEqual(body_data["time_step"].tolist(), [2, 3, 4])

    def test_transform_instantaneous(self):
        press_data = pd.DataFrame({"time_step": [1, 2, 3, 4, 5], "0": [1, 1, 1, 1, 1]})
        body_data = pd.DataFrame({"time_step": [1, 2, 3, 4, 5], "0": [1.1, 1.2, 1.3, 1.4, 1.5]})

        transformed_data = transform_to_cp(press_data, body_data, 0.05, 1, "instantaneous")
        self.assertIn("0", transformed_data.columns)
        self.assertEqual(len(transformed_data.iloc[0]), len(body_data.iloc[0]))

    def test_transform_average(self):
        press_data = pd.DataFrame({"time_step": [1, 2, 3, 4, 5], "0": [1, 1, 1, 1, 1]})
        body_data = pd.DataFrame({"time_step": [1, 2, 3, 4, 5], "0": [1.1, 1.2, 1.3, 1.4, 1.5]})

        transformed_data = transform_to_cp(press_data, body_data, 0.05, 1, "average")
        self.assertIn("0", transformed_data.columns)
        self.assertEqual(len(transformed_data.iloc[0]), len(body_data.iloc[0]))


if __name__ == "__main__":
    unittest.main()
