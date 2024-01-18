import math
import pathlib
import unittest

import numpy as np
import pandas as pd
from lnas import LnasGeometry

from cfdmod.api.geometry.transformation_config import TransformationConfig
from cfdmod.use_cases.pressure.chunking import process_timestep_groups, split_into_chunks
from cfdmod.use_cases.pressure.geometry import GeometryData, tabulate_geometry_data
from cfdmod.use_cases.pressure.zoning.zoning_model import ZoningModel


class TestChunking(unittest.TestCase):
    def setUp(self):
        time_series_data = {
            "time_step": np.repeat(range(10), 10),
            "value": range(100),
            "point_idx": np.tile(range(10), 10),
        }
        self.sample_df = pd.DataFrame(time_series_data)
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
        time_arr = self.sample_df.time_step.unique()
        step = math.ceil(len(time_arr) / self.number_of_chunks)
        expected_keys = [
            f"/range_{int(time_arr[i * step])}_{int(min((i + 1) * step - 1, len(time_arr) - 1))}"
            for i in range(self.number_of_chunks)
        ]

        split_into_chunks(self.sample_df, self.number_of_chunks, self.output_path)
        self.assertTrue(self.output_path.exists())

        with pd.HDFStore(self.output_path, "r") as store:
            self.assertEqual(len(store.keys()), self.number_of_chunks)
            self.assertListEqual(list(store.keys()), expected_keys)

        self.output_path.unlink()

    def test_process_timestep_groups(self):
        def mock_processing_function(
            cp_df: pd.DataFrame, _geom_df: pd.DataFrame, _geom: LnasGeometry
        ) -> pd.DataFrame:
            cp_df["value"] *= 2
            return cp_df

        split_into_chunks(self.sample_df, self.number_of_chunks, self.output_path)

        zoning = ZoningModel(x_intervals=[0, 5, 10])
        zoning.offset_limits(0.1)
        geometry = LnasGeometry(
            vertices=np.array([[0, 0, 0], [0, 10, 0], [10, 0, 0], [10, 10, 0]]),
            triangles=np.array([[0, 1, 2], [2, 1, 3]]),
        )

        geom_dict = {
            "sfc1": GeometryData(
                mesh=geometry, zoning_to_use=zoning, triangles_idxs=np.array([0, 1])
            )
        }
        geometry_df = tabulate_geometry_data(
            geom_dict=geom_dict,
            mesh_areas=geometry.areas,
            mesh_normals=geometry.normals,
            transformation=TransformationConfig(),
        )

        result_df = process_timestep_groups(
            self.output_path, geometry_df, geometry, mock_processing_function
        )

        self.assertTrue((result_df.value.to_numpy() == self.sample_df.value.to_numpy() * 2).all())
        self.assertTrue(result_df.time_step.nunique() == self.sample_df.time_step.nunique())

        self.output_path.unlink()


if __name__ == "__main__":
    unittest.main()
