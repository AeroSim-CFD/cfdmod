import pathlib

import numpy as np
import pandas as pd
import pytest
from lnas import LnasGeometry
from vtk.util.numpy_support import vtk_to_numpy  # type: ignore

from cfdmod.api.vtk.write_vtk import create_polydata_for_cell_data, write_polydata


@pytest.fixture()
def vertices():
    yield np.array([[0, 0, 0], [0, 10, 0], [10, 0, 0], [10, 10, 0]])


@pytest.fixture()
def triangles():
    yield np.array([[0, 1, 2], [1, 3, 2]])


@pytest.fixture()
def mock_data():
    yield pd.DataFrame({"point_idx": [0, 1], "scalar": np.array([0.1, 0.2], dtype=np.float32)})


def test_create_polydata(vertices, triangles, mock_data):
    mock_mesh = LnasGeometry(vertices, triangles)
    polydata = create_polydata_for_cell_data(data=mock_data, mesh=mock_mesh)
    data = vtk_to_numpy(polydata.GetCellData().GetArray("scalar"))

    assert polydata.GetNumberOfCells() == 2
    assert (data == mock_data["scalar"].to_numpy()).all()


def test_write_polydata(vertices, triangles, mock_data):
    output_filename = pathlib.Path("./output/vtk_test.vtp")
    mock_mesh = LnasGeometry(vertices, triangles)

    polydata = create_polydata_for_cell_data(data=mock_data, mesh=mock_mesh)
    write_polydata(output_filename, polydata)

    assert output_filename.exists()
