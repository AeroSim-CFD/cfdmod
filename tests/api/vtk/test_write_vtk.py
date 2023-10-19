import pathlib
import unittest

import numpy as np
import pandas as pd
from nassu.lnas import LagrangianGeometry
from vtk.util.numpy_support import vtk_to_numpy  # type: ignore

from cfdmod.api.vtk.write_vtk import create_polydata_for_cell_data, write_polydata


class TestWritePolydata(unittest.TestCase):
    def setUp(self):
        self.vertices = np.array([[0, 0, 0], [0, 10, 0], [10, 0, 0], [10, 10, 0]])
        self.triangles = np.array([[0, 1, 2], [1, 3, 2]])
        self.mock_data = pd.DataFrame(
            {"point_idx": [0, 1], "scalar": np.array([0.1, 0.2], dtype=np.float32)}
        )

    def test_create_polydata(self):
        mock_mesh = LagrangianGeometry(self.vertices, self.triangles)
        polydata = create_polydata_for_cell_data(data=self.mock_data, mesh=mock_mesh)
        data = vtk_to_numpy(polydata.GetCellData().GetArray("scalar"))

        self.assertEqual(polydata.GetNumberOfCells(), 2)
        self.assertTrue((data == self.mock_data["scalar"].to_numpy()).all())

    def test_write_polydata(self):
        output_filename = pathlib.Path("./output/vtk_test.vtp")
        mock_mesh = LagrangianGeometry(self.vertices, self.triangles)

        polydata = create_polydata_for_cell_data(data=self.mock_data, mesh=mock_mesh)
        write_polydata(output_filename, polydata)

        self.assertTrue(output_filename.exists())


if __name__ == "__main__":
    unittest.main()
