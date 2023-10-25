import pathlib
import unittest

import numpy as np

from cfdmod.api.vtk.probe_vtm import create_line, get_array_from_filter, probe_over_line, read_vtm
from cfdmod.use_cases.s1.probe import S1Probe
from cfdmod.use_cases.s1.profile import Profile


class TestS1Profile(unittest.TestCase):
    def test_vtm_profiles(self):
        field_data_path = pathlib.Path("./fixtures/tests/s1/vtm/example.vtm")
        pitot_probe = S1Probe(p1=[10, 10, 1], p2=[10, 10, 50], numPoints=100)
        example_probe = S1Probe(p1=[100, 10, 1], p2=[100, 10, 50], numPoints=100)

        reader = read_vtm(field_data_path)

        pitot_line = create_line(pitot_probe.p1, pitot_probe.p2, pitot_probe.numPoints - 1)
        probe_line = create_line(example_probe.p1, example_probe.p2, example_probe.numPoints - 1)

        pitot_filter = probe_over_line(pitot_line, reader)
        probe_filter = probe_over_line(probe_line, reader)

        pitot_data = get_array_from_filter(pitot_filter, array_lbl="ux")
        probe_data = get_array_from_filter(probe_filter, array_lbl="ux")

        pitot_heights = np.linspace(pitot_probe.p1[2], pitot_probe.p2[2], pitot_probe.numPoints)
        probe_heights = np.linspace(example_probe.p1[2], example_probe.p2[2], example_probe.numPoints)

        pitot_profile = Profile(heights=pitot_heights, values=pitot_data, label="Pitot")
        probe_profile = Profile(heights=probe_heights, values=probe_data, label="Example probe")

        s1_profile = probe_profile / pitot_profile

        self.assertEqual(s1_profile.heights.shape, s1_profile.values.shape)
        self.assertEqual(probe_profile.heights.shape, probe_profile.values.shape)
        self.assertEqual(pitot_profile.heights.shape, pitot_profile.values.shape)

    def test_csv_profiles(self):
        probes_path = pathlib.Path("./fixtures/tests/s1/csv")
        pitot_path = pathlib.Path("./fixtures/tests/s1/csv/Pitot-000_U.csv")
        pitot = Profile.from_csv(pitot_path, "z", "U_0", "Pitot")

        for file in [
            f for f in probes_path.iterdir() if "Pitot" not in f.name and f.name.endswith(".csv")
        ]:
            probe_path = pathlib.Path(file)
            probe = Profile.from_csv(probe_path, "z", "U_0", file.name)
            s1 = probe / pitot

            self.assertEqual(s1.heights.shape, s1.values.shape)
            self.assertEqual(probe.heights.shape, probe.values.shape)
        self.assertEqual(pitot.heights.shape, pitot.values.shape)


if __name__ == "__main__":
    unittest.main()
