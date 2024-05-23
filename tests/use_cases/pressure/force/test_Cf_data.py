import unittest

import numpy as np
import pandas as pd
from lnas import LnasFormat, LnasGeometry

from cfdmod.api.geometry.transformation_config import TransformationConfig
from cfdmod.use_cases.pressure.force.Cf_data import get_representative_areas, transform_Cf
from cfdmod.use_cases.pressure.geometry import GeometryData, tabulate_geometry_data
from cfdmod.use_cases.pressure.statistics import BasicStatisticModel
from cfdmod.use_cases.pressure.zoning.processing import calculate_statistics
from cfdmod.use_cases.pressure.zoning.zoning_model import ZoningModel
from cfdmod.utils import convert_dataframe_into_matrix


class TestCfData(unittest.TestCase):
    def setUp(self):
        self.cp_data = pd.DataFrame(
            {
                "point_idx": [0, 0, 0, 1, 1, 1],
                "cp": [0.1, 0.2, 0.3, 0.4, 0.5, 0.6],
                "time_normalized": [0, 1, 2, 0, 1, 2],
            }
        )

        vertices = np.array([[0, 0, 0], [10, 0, 0], [0, 10, 0], [10, 10, 0]])
        triangles = np.array([[0, 1, 2], [1, 3, 2]])

        self.body_geom = LnasGeometry(vertices, triangles)

    def test_get_representative_areas(self):
        tri_1_area = get_representative_areas(self.body_geom, np.array([0]))
        tri_2_area = get_representative_areas(self.body_geom, np.array([1]))

        self.assertEqual(tri_1_area, tri_2_area, np.array([0, 0, 100]))

    def test_transform_to_Cf(self):
        geom_data = GeometryData(
            mesh=self.body_geom, zoning_to_use=ZoningModel(), triangles_idxs=np.array([0, 1])
        )
        geometry_dict = {"body": geom_data}

        geometry_df = tabulate_geometry_data(
            geom_dict=geometry_dict,
            mesh_areas=self.body_geom.areas,
            mesh_normals=self.body_geom.normals,
            transformation=TransformationConfig(),
        )
        cf_data = transform_Cf(self.cp_data, geometry_df, self.body_geom)

        self.assertEqual(
            len(cf_data), self.cp_data.time_normalized.nunique() * geometry_df.region_idx.nunique()
        )  # Three timesteps x 1 region
        self.assertTrue(all([f"Cf{var}" in cf_data.columns for var in ["x", "y", "z"]]))

    def test_liquid_coefficients(self):
        cp_data = pd.DataFrame(
            {
                "point_idx": [0, 0, 0, 1, 1, 1, 2, 2, 2, 3, 3, 3],
                "cp": [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.1, 1.2],
                "time_normalized": [0, 1, 2, 0, 1, 2, 0, 1, 2, 0, 1, 2],
            }
        )

        upper_mesh = LnasFormat(
            version="", geometry=self.body_geom, surfaces={"sfc1": np.array([0, 1])}
        )
        lower_mesh = upper_mesh.copy()
        geometry_negative = lower_mesh.geometry
        geometry_negative.triangles = np.flip(geometry_negative.triangles, 1)
        geometry_negative._full_update()
        geometry_negative.vertices = (
            geometry_negative.vertices + geometry_negative.vertices_normals * 0.1
        )
        upper_mesh.join(lnas_fmts=[lower_mesh], surfaces_suffixes=["_lower"])
        geom_data = GeometryData(
            mesh=upper_mesh.geometry,
            zoning_to_use=ZoningModel(),
            triangles_idxs=np.array([0, 1, 2, 3]),
        )
        geometry_dict = {"body": geom_data}

        geometry_df = tabulate_geometry_data(
            geom_dict=geometry_dict,
            mesh_areas=upper_mesh.geometry.areas,
            mesh_normals=upper_mesh.geometry.normals,
            transformation=TransformationConfig(),
        )
        cf_data = transform_Cf(cp_data, geometry_df, upper_mesh.geometry)
        cf_data = convert_dataframe_into_matrix(
            cf_data,
            row_data_label="time_normalized",
            column_data_label="region_idx",
            value_data_label="Cfz",
        )
        cf_stats = calculate_statistics(
            historical_data=cf_data,
            statistics_to_apply=[
                BasicStatisticModel(stats="mean"),
                BasicStatisticModel(stats="rms"),
                BasicStatisticModel(stats="skewness"),
                BasicStatisticModel(stats="kurtosis"),
            ],
        )
        self.assertEqual(round(cf_stats.iloc[0]["mean"], 1), 0.6)
