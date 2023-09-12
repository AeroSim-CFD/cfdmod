import pathlib
import tempfile
import unittest

import numpy as np
import trimesh

from cfdmod.use_cases.altimetry import (
    AltimetrySection,
    AltimetryShed,
    plot_altimetry_profiles,
    plot_profiles,
    plot_surface,
)


class TestAltimetryUseCase(unittest.TestCase):
    def test_image_generation(self):
        temp_dir = tempfile.TemporaryDirectory()
        output_path = pathlib.Path(temp_dir.name)
        surface_mesh: trimesh.Trimesh = trimesh.load_mesh("./fixtures/tests/altimetry/terrain.stl")

        shed_start = np.array([-50, -50, 820])
        shed_end = np.array([50, 50, 820])

        shed = AltimetryShed(shed_start, shed_end)

        altimetry_section = AltimetrySection.from_points("example", shed_start, shed_end)
        altimetry_section.slice_surface(surface_mesh)
        altimetry_section.include_shed(shed)

        plot_surface(surface_mesh, [altimetry_section], output_path)
        plot_profiles([altimetry_section], output_path)
        plot_altimetry_profiles(altimetry_section, output_path)

        self.assertEqual(len([x for x in output_path.glob("**/*") if x.is_file()]), 3)
        temp_dir.cleanup()

    def test_slicing(self):
        surface_mesh: trimesh.Trimesh = trimesh.load_mesh("./fixtures/tests/altimetry/terrain.stl")

        plane_normal = np.array([1, 0, 0])
        plane_origin = np.array([0, 0, 820])

        altimetry_section = AltimetrySection("example", plane_origin, plane_normal)
        altimetry_section.slice_surface(surface_mesh)

        self.assertNotEqual(len(altimetry_section.section_vertices.x), 0)
        self.assertNotEqual(len(altimetry_section.section_vertices.y), 0)
        self.assertNotEqual(len(altimetry_section.section_vertices.z), 0)


if __name__ == "__main__":
    unittest.main()
