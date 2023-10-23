import unittest

import numpy as np
import pandas as pd
from nassu.lnas import LagrangianGeometry

from cfdmod.use_cases.pressure.moment.Cm_data import (
    get_lever_relative_position_df,
    get_representative_volume,
    transform_to_Cm,
)


class TestCfData(unittest.TestCase):
    def setUp(self):
        self.body_data = pd.DataFrame(
            {
                "cp": [0.1, 0.2, 0.3, 0.4],
                "Ax": [1, 1, 1, 1],
                "Ay": [2, 2, 2, 2],
                "Az": [3, 3, 3, 3],
                "region_idx": [0, 0, 0, 0],
                "time_step": [0, 0, 1, 1],
            }
        )

        vertices = np.array([[0, 0, 0], [10, 0, 0], [0, 10, 0], [10, 10, 0]])
        triangles = np.array([[0, 1, 2], [1, 3, 2]])

        self.body_geom = LagrangianGeometry(vertices, triangles)

    def test_transform_to_Cf(self):
        centroids = np.mean(self.body_geom.triangle_vertices, axis=1)
        position_df = get_lever_relative_position_df(
            centroids=centroids, lever_origin=(0, 0, 0), geometry_idx=np.array([0, 1])
        )
        body_data = pd.merge(self.body_data, position_df, on="point_idx", how="left")
        transformed_data = transform_to_Cm(body_data, self.body_geom)
        self.assertIsNotNone(transformed_data)
        # Add more assertions to check the actual data

    def test_get_representative_areas(self):
        V_rep = get_representative_volume(self.body_geom)
        self.assertIsNotNone(V_rep)
        # Add more assertions to check the actual data
