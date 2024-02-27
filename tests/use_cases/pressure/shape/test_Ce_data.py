import unittest

import numpy as np
import pandas as pd
from lnas import LnasGeometry

from cfdmod.api.geometry.transformation_config import TransformationConfig
from cfdmod.use_cases.pressure.geometry import GeometryData, tabulate_geometry_data
from cfdmod.use_cases.pressure.shape.Ce_data import calculate_statistics, transform_Ce
from cfdmod.use_cases.pressure.statistics import Statistics
from cfdmod.use_cases.pressure.zoning.zoning_model import ZoningModel


class TestCeData(unittest.TestCase):
    def setUp(self):
        self.mesh = LnasGeometry(
            vertices=np.array([[0, 0, 0], [0, 10, 0], [10, 0, 0], [10, 10, 0]]),
            triangles=np.array([[0, 1, 2], [2, 1, 3]]),
        )
        self.cp_data = pd.DataFrame(
            {
                "point_idx": [0, 0, 0, 1, 1, 1],
                "cp": [0.1, 0.2, 0.3, 0.4, 0.5, 0.6],
                "time_step": [0, 1, 2, 0, 1, 2],
            }
        )
        self.region_data = pd.DataFrame(
            {
                "region_idx": [0, 0, 0, 0, 1, 1, 1, 1],
                "Ce": [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8],
            }
        )
        self.zoning = ZoningModel(x_intervals=[0, 5, 10])
        self.zoning.offset_limits(0.1)
        self.statistics_to_apply: list[Statistics] = [
            "mean",
            "min",
            "max",
            "std",
            "skewness",
            "kurtosis",
        ]

    def test_transform_Ce(self):
        geom_dict = {
            "sfc1": GeometryData(
                mesh=self.mesh, zoning_to_use=self.zoning, triangles_idxs=np.array([0, 1])
            )
        }
        geometry_df = tabulate_geometry_data(
            geom_dict,
            mesh_areas=self.mesh.areas,
            mesh_normals=self.mesh.normals,
            transformation=TransformationConfig(),
        )
        ce_data = transform_Ce(self.cp_data, geometry_df, self.mesh)

        self.assertEqual(
            len(ce_data), self.cp_data.time_step.nunique() * self.cp_data.point_idx.nunique()
        )  # Three timesteps x 2 triangle
        self.assertTrue("Ce" in ce_data.columns)

    def test_calculate_statistics(self):
        result = calculate_statistics(
            historical_data=self.region_data,
            statistics_to_apply=self.statistics_to_apply,
            variables=["Ce"],
            group_by_key="region_idx",
        )

        self.assertEqual(len(result), 2)  # Two regions (0, 1)
        self.assertTrue((result.isnull().sum() == 0).all())


if __name__ == "__main__":
    unittest.main()
