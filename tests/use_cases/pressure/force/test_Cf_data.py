import unittest

import numpy as np
import pandas as pd
from lnas import LnasGeometry

from cfdmod.use_cases.pressure.force.Cf_data import transform_to_Cf


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

        self.body_geom = LnasGeometry(vertices, triangles)

    def test_transform_to_Cf(self):
        transformed_data = transform_to_Cf(self.body_data, self.body_geom)
        self.assertIsNotNone(transformed_data)
