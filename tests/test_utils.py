import unittest

import numpy as np
import pandas as pd

from cfdmod.utils import convert_dataframe_into_matrix, convert_matrix_into_dataframe


class TestUtils(unittest.TestCase):
    def setUp(self):
        n_points, n_timesteps = 200, 100
        df_dict = {
            "point_idx": np.tile(np.array(range(n_points), dtype=np.int32), n_timesteps),
            "rho": np.random.uniform(0.99, 1.01, n_timesteps * n_points),
            "time_step": np.repeat(np.linspace(0, 100, n_timesteps), n_points),
        }
        self.dataframe_source = pd.DataFrame(data=df_dict)

    def test_conversion(self):
        matrix_from_dataframe = convert_dataframe_into_matrix(self.dataframe_source)
        dataframe_from_matrix = convert_matrix_into_dataframe(matrix_from_dataframe)
        matrix_from_converted_dataframe = convert_dataframe_into_matrix(dataframe_from_matrix)

        self.assertTrue(dataframe_from_matrix.equals(self.dataframe_source))
        self.assertTrue(matrix_from_converted_dataframe.equals(matrix_from_dataframe))
