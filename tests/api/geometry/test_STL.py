import pathlib
import unittest

import numpy as np

from cfdmod.api.geometry.STL import export_stl, read_stl


class TestSTL(unittest.TestCase):
    def test_STL_example(self):
        mesh_path = pathlib.Path("./output/api/geometry/test_square.stl")

        single_square_verts = np.array(
            [[0, 0, 0], [0, 1, 0], [0, 0, 1], [0, 1, 1]], dtype=np.float32
        )
        single_square_triangles = np.array([[0, 1, 2], [2, 1, 3]], dtype=np.uint32)

        export_stl(mesh_path, single_square_verts, single_square_triangles)

        file_verts, file_tri = read_stl(mesh_path)

        self.assertTrue(all((single_square_verts == file_verts).reshape(1, 12)[0]))
        self.assertTrue(all((single_square_triangles == file_tri).reshape(1, 6)[0]))


if __name__ == "__main__":
    unittest.main()
