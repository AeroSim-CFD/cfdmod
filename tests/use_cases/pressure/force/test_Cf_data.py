import unittest

import numpy as np
import pandas as pd
from nassu.lnas import LagrangianGeometry

from cfdmod.use_cases.pressure.force.Cf_data import get_representative_areas, transform_to_Cf


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
        transformed_data = transform_to_Cf(self.body_data, self.body_geom)
        self.assertIsNotNone(transformed_data)
        # Add more assertions to check the actual data

    def test_get_representative_areas(self):
        Ax, Ay, Az = get_representative_areas(self.body_geom)
        self.assertIsNotNone(Ax)
        self.assertIsNotNone(Ay)
        self.assertIsNotNone(Az)
        # Add more assertions to check the actual data
