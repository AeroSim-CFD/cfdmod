import unittest

import numpy as np
import pandas as pd
from lnas import LnasGeometry

from cfdmod.api.geometry.transformation_config import TransformationConfig
from cfdmod.use_cases.pressure.geometry import GeometryData, tabulate_geometry_data
from cfdmod.use_cases.pressure.moment.Cm_data import (
    add_lever_arm_to_geometry_df,
    get_representative_volume,
    transform_Cm,
)
from cfdmod.use_cases.pressure.zoning.zoning_model import ZoningModel


class TestCmData(unittest.TestCase):
    def setUp(self):
        self.body_data = pd.DataFrame(
            {
                "cp": [0.1, 0.2, 0.3, 0.4],
                "time_normalized": [0, 0, 1, 1],
                "point_idx": [0, 1, 0, 1],
            }
        )

        vertices = np.array([[0, 0, 0], [10, 0, 0], [0, 10, 0], [10, 10, 0]])
        triangles = np.array([[0, 1, 2], [1, 3, 2]])

        self.body_geom = LnasGeometry(vertices, triangles)
        self.geom_data = GeometryData(
            mesh=self.body_geom, zoning_to_use=ZoningModel(), triangles_idxs=np.array([0, 1])
        )
        geometry_dict = {"body": self.geom_data}
        self.geometry_df = tabulate_geometry_data(
            geom_dict=geometry_dict,
            mesh_areas=self.body_geom.areas,
            mesh_normals=self.body_geom.normals,
            transformation=TransformationConfig(),
        )

    def test_add_lever_arm(self):
        geometry_df = add_lever_arm_to_geometry_df(
            geom_data=self.geom_data,
            transformation=TransformationConfig(),
            lever_origin=[0, 0, 10],
            geometry_df=self.geometry_df,
        )

        self.assertTrue(all([f"r{dir}" in geometry_df.columns for dir in ["x", "y", "z"]]))

    def test_transform_Cm(self):
        geometry_df = add_lever_arm_to_geometry_df(
            geom_data=self.geom_data,
            transformation=TransformationConfig(),
            lever_origin=[0, 0, 10],
            geometry_df=self.geometry_df,
        )
        Cm_data = transform_Cm(
            raw_cp=self.body_data, geometry_df=geometry_df, geometry=self.body_geom
        )

        self.assertIsNotNone(Cm_data)
        self.assertTrue(all([f"Cm{dir}" in Cm_data.columns for dir in ["x", "y", "z"]]))

    def test_get_representative_volume(self):
        V_rep = get_representative_volume(
            self.body_geom, np.arange(0, len(self.body_geom.triangles))
        )
        shifted_geom = self.body_geom.copy()
        shifted_geom.vertices[-1][2] = 10  # Shifted z coord for the last vertex
        shifted_geom._full_update()
        shifted_V_rep = get_representative_volume(
            shifted_geom, np.arange(0, len(self.body_geom.triangles))
        )

        self.assertEqual(V_rep, 100)
        self.assertEqual(shifted_V_rep, 1000)
