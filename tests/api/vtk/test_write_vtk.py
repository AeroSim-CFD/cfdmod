import pathlib
import unittest

from vtk import vtkCellArray, vtkPoints, vtkPolyData

from cfdmod.api.vtk.write_vtk import write_polydata


class TestWritePolydata(unittest.TestCase):
    def test_write_polydata(self):
        output_filename = pathlib.Path("output.vtp")
        points = vtkPoints()
        vertices = vtkCellArray()
        poly_data = vtkPolyData()
        poly_data.SetPoints(points)
        poly_data.SetVerts(vertices)

        write_polydata(output_filename, poly_data)

        self.assertTrue(output_filename.exists())
        output_filename.unlink()


if __name__ == "__main__":
    unittest.main()
