import unittest

import matplotlib.tri as tri
import numpy as np

from cfdmod.use_cases.loft.functions import (
    find_border,
    generate_circular_loft_vertices,
    generate_loft_triangles,
    get_angle_between,
    project_border,
)


class TestLoftFunctions(unittest.TestCase):
    def setUp(self):
        self.nx, self.ny = 3, 3
        x = np.linspace(-10, 10, self.nx + 1)
        y = np.linspace(-10, 10, self.ny + 1)
        z = np.array([1])
        xv, yv, zv = np.meshgrid(x, y, z)

        vertices = np.vstack((xv.flatten(), yv.flatten(), zv.flatten())).T
        triangles = tri.Triangulation(vertices[:, 0], vertices[:, 1]).triangles

        self.triangle_vertices = vertices[triangles]
        super().setUp()

    def test_find_border(self):
        border_verts, border_edges = find_border(triangle_vertices=self.triangle_vertices)
        expected_edge_count = (self.nx + self.ny) * 2
        expected_vertex_count = (self.nx + self.ny + 2) * 2 - 4
        self.assertEqual(len(border_edges), expected_edge_count)
        self.assertEqual(len(border_verts), expected_vertex_count)

    def test_angle_between(self):
        vec1 = np.array([1, 0, 0])
        vec2 = np.array([0, 1, 0])
        vec3 = np.array([1, 1, 0])
        vec4 = np.array([-1, -1, 0])
        self.assertEqual(get_angle_between(vec1, vec2), 90)
        self.assertEqual(get_angle_between(vec1, vec3), 45)
        self.assertEqual(get_angle_between(vec1, vec4), 225)
        self.assertEqual(get_angle_between(vec2, vec3), 315)
        self.assertEqual(get_angle_between(vec3, vec4), 180)

    def test_project_border(self):
        border_verts, _ = find_border(triangle_vertices=self.triangle_vertices)
        border_profile, _ = project_border(border_verts, projection_diretion=np.array([1, 0, 0]))
        self.assertTrue(all(border_profile[:, 0] >= 0))
        border_profile, _ = project_border(border_verts, projection_diretion=np.array([-1, 0, 0]))
        self.assertTrue(all(border_profile[:, 0] <= 0))
        border_profile, _ = project_border(border_verts, projection_diretion=np.array([0, 1, 0]))
        self.assertTrue(all(border_profile[:, 1] >= 0))
        border_profile, _ = project_border(border_verts, projection_diretion=np.array([0, -1, 0]))
        self.assertTrue(all(border_profile[:, 1] <= 0))

    def test_loft_surface(self):
        projection_direction = np.array([1, 0, 0])
        border_verts, _ = find_border(triangle_vertices=self.triangle_vertices)
        border_profile, center = project_border(
            border_verts, projection_diretion=projection_direction
        )
        loft_verts = generate_circular_loft_vertices(
            border_profile=border_profile,
            projection_diretion=projection_direction,
            loft_length=100,
            loft_z_pos=1,
            mesh_center=center,
        )
        loft_tri, loft_normals = generate_loft_triangles(
            border_profile=border_profile, loft_vertices=loft_verts
        )
        self.assertEqual(len(border_profile), len(loft_verts))
        self.assertEqual(len(border_profile) - 1, len(loft_verts) - 1, len(loft_tri) / 2)


if __name__ == "__main__":
    unittest.main()
