import pathlib
import unittest

import numpy as np

from cfdmod.api.vtk.probe_vtm import create_line, get_array_from_filter, probe_over_line, read_vtm


class TestProbeVTM(unittest.TestCase):
    def setUp(self):
        self.p1 = [10, 20, 1]
        self.p2 = [10, 20, 40]
        self.vtm_path = pathlib.Path("./fixtures/tests/s1/vtm/example.vtm")

    def test_create_line(self):
        line = create_line(self.p1, self.p2, numPoints=10)
        self.assertEqual(10, line.GetResolution())

    def test_probe_vtm(self):
        line = create_line(self.p1, self.p2, numPoints=10)
        reader = read_vtm(self.vtm_path)
        probe = probe_over_line(line, reader)
        data = get_array_from_filter(probe, array_lbl="ux")

        self.assertEqual(len(data), 11)


if __name__ == "__main__":
    unittest.main()
