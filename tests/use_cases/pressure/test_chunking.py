import math
import pathlib
import unittest

import numpy as np
import pandas as pd

from cfdmod.use_cases.pressure.chunking import join_chunks_for_points, split_into_chunks


class TestChunking(unittest.TestCase):
    def setUp(self):
        time_series_data = {
            "time_step": np.repeat(range(10), 10),
            "value": range(100),
            "point_idx": np.tile(range(10), 10),
        }
        self.sample_time_series_df = pd.DataFrame(time_series_data)
        self.number_of_chunks = 3
        self.output_path = pathlib.Path("./output/chunking.h5")

    def test_raises_error_if_dataframe_lacks_time_step_column(self):
        time_series_df = pd.DataFrame({"value": [10, 20, 30]})

        with self.assertRaises(ValueError) as context:
            split_into_chunks(time_series_df, self.number_of_chunks, self.output_path)

        self.assertEqual(
            str(context.exception),
            "Time series dataframe must have a time_step column to be chunked",
        )

    def test_split_into_chunks(self):
        time_arr = self.sample_time_series_df.time_step.unique()
        step = math.ceil(len(time_arr) / self.number_of_chunks)
        expected_keys = [
            f"/range_{int(time_arr[i * step])}_{int(min((i + 1) * step - 1, len(time_arr) - 1))}"
            for i in range(self.number_of_chunks)
        ]

        split_into_chunks(self.sample_time_series_df, self.number_of_chunks, self.output_path)
        self.assertTrue(self.output_path.exists())

        with pd.HDFStore(self.output_path, "r") as store:
            self.assertEqual(len(store.keys()), self.number_of_chunks)
            self.assertListEqual(list(store.keys()), expected_keys)

        self.output_path.unlink()

    def test_join_chunks_for_points(self):
        filtered_idx = np.array(range(5))

        split_into_chunks(self.sample_time_series_df, self.number_of_chunks, self.output_path)
        result_df = join_chunks_for_points(self.output_path, point_idxs=filtered_idx)

        self.assertTrue(all(filtered_idx == result_df.point_idx.unique()))


if __name__ == "__main__":
    unittest.main()
