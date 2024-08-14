import pathlib
import unittest

from cfdmod.use_cases.loft.parameters import LoftCaseConfig


class TestLoftFunctions(unittest.TestCase):
    def setUp(self):
        self.cfg = LoftCaseConfig.from_file(pathlib.Path("./fixtures/tests/loft/loft_params.yaml"))
        super().setUp()

    def test_inherit_from_default(self):
        default_cfg = self.cfg.cases["default"]
        inherited_cfg = self.cfg.cases["inherit_loft"]
        inherited_elem_size_cfg = self.cfg.cases["inherit_element_size"]

        self.assertEqual(inherited_cfg.mesh_element_size, default_cfg.mesh_element_size)
        self.assertEqual(inherited_cfg.wind_source_angle, default_cfg.wind_source_angle)
        self.assertEqual(inherited_cfg.upwind_elevation, default_cfg.upwind_elevation)
        self.assertNotEqual(inherited_cfg.loft_length, default_cfg.loft_length)

        self.assertEqual(inherited_elem_size_cfg.mesh_element_size, default_cfg.mesh_element_size)
        self.assertNotEqual(
            inherited_elem_size_cfg.wind_source_angle, default_cfg.wind_source_angle
        )
        self.assertNotEqual(inherited_elem_size_cfg.upwind_elevation, default_cfg.upwind_elevation)
        self.assertNotEqual(inherited_elem_size_cfg.loft_length, default_cfg.loft_length)


if __name__ == "__main__":
    unittest.main()
