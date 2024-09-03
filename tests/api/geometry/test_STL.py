import pathlib

import numpy as np

from cfdmod.api.geometry.STL import export_stl, read_stl


def test_STL_example():
    mesh_path = pathlib.Path("./output/api/geometry/test_square.stl")

    single_square_triangles = np.array(
        [[[0, 0, 0], [0, 1, 0], [0, 0, 1]], [[0, 1, 1], [0, 1, 0], [0, 0, 1]]],
        dtype=np.float32,
    )
    single_square_normals = np.array([[1, 0, 0], [1, 0, 0]], dtype=np.float32)

    export_stl(mesh_path, single_square_triangles, single_square_normals)

    file_triangles, file_normals = read_stl(mesh_path)

    assert all((single_square_triangles == file_triangles).reshape(1, 18)[0])
    assert all((single_square_normals == file_normals).reshape(1, 6)[0])
