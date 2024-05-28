import math
import pathlib
import unittest

import numpy as np
import pandas as pd
from lnas import LnasGeometry

from cfdmod.api.geometry.transformation_config import TransformationConfig
from cfdmod.use_cases.pressure.chunking import (
    HDFGroupInterface,
    process_timestep_groups,
    split_into_chunks,
)
from cfdmod.use_cases.pressure.geometry import GeometryData, tabulate_geometry_data
from cfdmod.use_cases.pressure.zoning.zoning_model import ZoningModel


def _mock_processing_function(
    cp_df: pd.DataFrame, _geom_df: pd.DataFrame, _geom: LnasGeometry
) -> pd.DataFrame:
    result = cp_df.copy()
    result["value"] *= 2

    return result


class TestChunking(unittest.TestCase):
    def setUp(self):
        time_series_data = {
            "time_step": np.repeat(range(10), 10),
            "value": range(100),
            "point_idx": np.tile(range(10), 10),
        }
        zoning = ZoningModel(x_intervals=[0, 5, 10])

        self.geometry = LnasGeometry(
            vertices=np.array([[0, 0, 0], [0, 10, 0], [10, 0, 0], [10, 10, 0]]),
            triangles=np.array([[0, 1, 2], [2, 1, 3]]),
        )
        self.sample_df = pd.DataFrame(time_series_data)
        self.number_of_chunks = 3
        self.output_path = pathlib.Path("./output/chunking.h5")
        self.zoning = zoning.offset_limits(0.1)

    def test_raises_error_if_dataframe_lacks_time_step_column(self):
        time_series_df = pd.DataFrame({"value": [10, 20, 30]})

        with self.assertRaises(ValueError) as context:
            split_into_chunks(time_series_df, self.number_of_chunks, self.output_path)

        self.assertEqual(
            str(context.exception),
            "Time series dataframe must have a time_step column to be chunked",
        )

    def test_split_more_than_possible(self):
        with self.assertRaises(ValueError) as context:
            # sample_df has 10 time_step values and it is impossible to split
            # into 6 chunks with at least two steps each
            split_into_chunks(self.sample_df, 6, self.output_path)

        self.assertEqual(str(context.exception), "There must be at least two steps in each chunk")

    def test_split_into_chunks(self):
        time_arr = self.sample_df.time_step.unique()
        step = math.ceil(len(time_arr) / self.number_of_chunks)
        expected_keys = [
            HDFGroupInterface.time_key(time_arr[i * step]) for i in range(self.number_of_chunks)
        ]

        split_into_chunks(self.sample_df, self.number_of_chunks, self.output_path)
        self.assertTrue(self.output_path.exists())

        with pd.HDFStore(self.output_path, "r") as store:
            self.assertEqual(len(store.keys()), self.number_of_chunks)
            self.assertListEqual(list(store.keys()), expected_keys)

        self.output_path.unlink()

    def test_process_timestep_groups(self):
        split_into_chunks(self.sample_df, self.number_of_chunks, self.output_path)

        geom_dict = {
            "sfc1": GeometryData(
                mesh=self.geometry, zoning_to_use=self.zoning, triangles_idxs=np.array([0, 1])
            )
        }
        geometry_df = tabulate_geometry_data(
            geom_dict=geom_dict,
            mesh_areas=self.geometry.areas,
            mesh_normals=self.geometry.normals,
            transformation=TransformationConfig(),
        )

        result_df = process_timestep_groups(
            self.output_path,
            geometry_df,
            self.geometry,
            _mock_processing_function,
            time_column_label="time_step",
        )
        self.sample_df.sort_values(by=["time_step", "point_idx"], inplace=True)
        result_df.sort_values(by=["time_step", "point_idx"], inplace=True)

        self.assertTrue((result_df.value.to_numpy() == self.sample_df.value.to_numpy() * 2).all())
        self.assertTrue(result_df.time_step.nunique() == self.sample_df.time_step.nunique())

        self.output_path.unlink()

    def test_process_groups_statistics(self):
        geom_dict = {
            "sfc1": GeometryData(
                mesh=self.geometry, zoning_to_use=self.zoning, triangles_idxs=np.array([0, 1])
            )
        }
        geometry_df = tabulate_geometry_data(
            geom_dict=geom_dict,
            mesh_areas=self.geometry.areas,
            mesh_normals=self.geometry.normals,
            transformation=TransformationConfig(),
        )
        self.sample_df.sort_values(by=["time_step", "point_idx"], inplace=True)

        split_into_chunks(self.sample_df, self.number_of_chunks, self.output_path)
        first_df = process_timestep_groups(
            self.output_path,
            geometry_df,
            self.geometry,
            _mock_processing_function,
            time_column_label="time_step",
        )
        avg = first_df.value.mean()
        std = first_df.value.std()
        min_val, max_val = first_df.value.min(), first_df.value.max()

        self.output_path.unlink()
        self.sample_df.sort_values(by=["time_step", "point_idx"], inplace=True)

        split_into_chunks(self.sample_df, self.number_of_chunks + 2, self.output_path)
        second_df = process_timestep_groups(
            self.output_path,
            geometry_df,
            self.geometry,
            _mock_processing_function,
            time_column_label="time_step",
        )

        self.assertEqual(avg, second_df.value.mean())
        self.assertEqual(std, second_df.value.std())
        self.assertEqual(min_val, second_df.value.min())
        self.assertEqual(max_val, second_df.value.max())

        self.output_path.unlink()


if __name__ == "__main__":
    unittest.main()
