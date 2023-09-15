import pathlib
import unittest

import numpy as np

from cfdmod.api.geometry.STL import export_stl, read_stl


class TestSTL(unittest.TestCase):
    def test_STL_file(self):
        mesh_path = pathlib.Path("./output/api/geometry/test_triangle.stl")

        single_tri_verts = np.array([[0, 0, 0], [0, 1, 0], [0, 0, 1]], dtype=np.float32)
        single_tri_triangles = np.array([[0, 1, 2]], dtype=np.uint32)

        export_stl(mesh_path, single_tri_verts, single_tri_triangles)

        file_verts, file_tri = read_stl(mesh_path)


if __name__ == "__main__":
    unittest.main()
