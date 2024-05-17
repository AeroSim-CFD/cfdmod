import unittest

from lnas import LnasFormat, LnasGeometry
from cfdmod.use_cases.pressure.shape.Ce_config import CeConfig

class TestCeGeometry(unittest.TestCase):
    def setUp(self):
        self.mesh = LnasFormat(version="", geometry=LnasGeometry(vertices=, triangles=), surfaces={})
        self.cfg = CeConfig(statistics=[], zoning=, sets={}, transformation=)
        self.surface_dict = {}
        
    def test_get_geometry_data(self):
        return

    def test_generate_regions_mesh(self):
        return


if __name__ == "__main__":
    unittest.main()
