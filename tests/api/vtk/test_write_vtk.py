import pathlib
import unittest
from unittest.mock import Mock, patch

from cfdmod.api.vtk.write_vtk import write_polydata


class TestWritePolydata(unittest.TestCase):
    @patch("cfdmod.api.vtk.write_vtk.vtkXMLPolyDataWriter")
    @patch("cfdmod.api.vtk.write_vtk.create_folders_for_file")
    def test_write_polydata(self, MockCreateFolders, MockWriter):
        output_filename = pathlib.Path("output.vtp")
        poly_data = Mock()

        write_polydata(output_filename, poly_data)

        MockCreateFolders.assert_called_with(output_filename)
        MockWriter().SetFileName.assert_called_with(output_filename.as_posix())
        MockWriter().SetInputData.assert_called_with(poly_data)
        MockWriter().Write.assert_called()


if __name__ == "__main__":
    unittest.main()
