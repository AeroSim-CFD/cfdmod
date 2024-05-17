import unittest

import numpy as np
from lnas import LnasFormat, LnasGeometry

from cfdmod.api.geometry.transformation_config import TransformationConfig
from cfdmod.use_cases.pressure.shape.Ce_config import CeConfig, ZoningConfig
from cfdmod.use_cases.pressure.shape.Ce_data import get_surface_dict
from cfdmod.use_cases.pressure.shape.Ce_geom import generate_regions_mesh, get_geometry_data
from cfdmod.use_cases.pressure.statistics import BasicStatisticModel
from cfdmod.use_cases.pressure.zoning.zoning_model import ZoningModel


class TestCeGeometry(unittest.TestCase):
    def setUp(self):
        self.zoning = ZoningModel(x_intervals=[0, 5, 10], y_intervals=[0, 10], z_intervals=[0, 10])
        self.zoning.offset_limits(0.1)
        geom = LnasGeometry(
            vertices=np.array([[0, 0, 0], [0, 10, 0], [10, 0, 0], [10, 10, 0]]),
            triangles=np.array([[0, 1, 2], [2, 1, 3]]),
        )
        self.mesh = LnasFormat(
            version="", geometry=geom, surfaces={"sfc1": np.array([0]), "sfc2": np.array([1])}
        )
        self.cfg = CeConfig(
            statistics=[
                BasicStatisticModel(stats="mean"),
                BasicStatisticModel(stats="rms"),
                BasicStatisticModel(stats="skewness"),
                BasicStatisticModel(stats="kurtosis"),
            ],
            zoning=ZoningConfig(global_zoning=self.zoning),
            sets={},
            transformation=TransformationConfig(),
        )
        self.surface_dict = {}

    def test_get_geometry_data(self):
        sfc_dict = get_surface_dict(cfg=self.cfg, mesh=self.mesh)
        geometry_dict = get_geometry_data(surface_dict=sfc_dict, cfg=self.cfg, mesh=self.mesh)

        self.assertEqual(len(geometry_dict), len(sfc_dict), len(self.mesh.surfaces.keys()))

    def test_get_surface_dict(self):
        sfc_dict = get_surface_dict(cfg=self.cfg, mesh=self.mesh)

        self.assertTrue([k in sfc_dict.keys() for k in self.mesh.surfaces.keys()])
        self.assertTrue([k in self.mesh.surfaces.values() for k in sfc_dict.values()])

    def test_generate_regions_mesh(self):
        sfc_dict = get_surface_dict(cfg=self.cfg, mesh=self.mesh)
        geometry_dict = get_geometry_data(surface_dict=sfc_dict, cfg=self.cfg, mesh=self.mesh)
        for geometry_data in geometry_dict.values():
            regions_mesh, regions_mesh_triangles_indexing = generate_regions_mesh(
                geom_data=geometry_data, cfg=self.cfg
            )

            self.assertEqual(len(regions_mesh_triangles_indexing), len(regions_mesh.triangles))


if __name__ == "__main__":
    unittest.main()
