import unittest

import numpy as np
from nassu.lnas import LagrangianFormat, LagrangianGeometry

from cfdmod.use_cases.pressure.geometry import get_excluded_surfaces


class TestGeometry(unittest.TestCase):
    def setUp(self):
        vertices = np.array([[0, 0, 0], [0, 10, 0], [10, 0, 0], [10, 10, 0]])
        triangles = np.array([[0, 1, 2], [1, 3, 2]])
        geometry = LagrangianGeometry(vertices=vertices, triangles=triangles)
        self.mesh = LagrangianFormat(
            version="",
            name="mock mesh",
            normalization=None,
            geometry=geometry,
            surfaces={"sfc1": np.array([0]), "sfc2": np.array([1])},
        )

    def test_no_excluded_surfaces(self):
        sfc_list = []
        with self.assertRaises(Exception) as context:
            get_excluded_surfaces(self.mesh, sfc_list)
        self.assertEqual(
            str(context.exception), "No geometry could be filtered from the list of surfaces."
        )

    def test_some_excluded_surfaces(self):
        sfc_list = ["sfc2"]
        result = get_excluded_surfaces(self.mesh, sfc_list)
        self.assertIsInstance(result, LagrangianGeometry)  # Expecting a LagrangianGeometry object

    def test_excluded_surface_not_in_mesh(self):
        sfc_list = ["sfc3"]
        with self.assertRaises(Exception) as context:
            get_excluded_surfaces(self.mesh, sfc_list)
        self.assertEqual(str(context.exception), "Surface is not defined in LNAS.")


if __name__ == "__main__":
    unittest.main()
