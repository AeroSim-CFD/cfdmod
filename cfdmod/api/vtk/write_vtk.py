import pathlib
from typing import Sequence

import pandas as pd
from lnas import LnasGeometry
from vtk import (
    vtkAppendPolyData,
    vtkCellArray,
    vtkFloatArray,
    vtkIdList,
    vtkPoints,
    vtkPolyData,
    vtkXMLPolyDataWriter,
)

from cfdmod.utils import create_folders_for_file


def _mkVtkIdList(it) -> vtkIdList:
    """Makes a vtkIdList from a Python iterable. I'm kinda surprised that
     this is necessary, since I assumed that this kind of thing would
     have been built into the wrapper and happen transparently, but it
     seems not.

    Args:
        it (Iterable): A python iterable.

    Returns:
        vtkIdList: A vtkIdList
    """
    vil = vtkIdList()
    for i in it:
        vil.InsertNextId(int(i))
    return vil


def create_polydata_for_cell_data(data: pd.DataFrame, mesh: LnasGeometry) -> vtkPolyData:
    """Creates a vtkPolyData for cell data combined with mesh description

    Args:
        data (pd.DataFrame): Compiled cell data
        mesh (LnasGeometry): Mesh description

    Returns:
        vtkPolyData: Extracted polydata
    """
    # We'll create the building blocks of polydata including data attributes.
    polyData = vtkPolyData()
    points = vtkPoints()
    polys = vtkCellArray()

    # Load the point, cell, and data attributes.
    for i, xi in enumerate(mesh.vertices):
        points.InsertPoint(i, xi)
    for pt in mesh.triangles:
        polys.InsertNextCell(_mkVtkIdList(pt))

    # We now assign the pieces to the vtkPolyData.
    polyData.SetPoints(points)
    polyData.SetPolys(polys)

    scalars = [c for c in data.columns if c != "point_idx"]

    for scalar_lbl in scalars:
        scalars = vtkFloatArray()
        scalars.SetName(scalar_lbl)
        for i, value in zip(data["point_idx"].to_numpy(), data[scalar_lbl].to_numpy()):
            scalars.InsertTuple1(i, value)
        polyData.GetCellData().AddArray(scalars)

    return polyData


def merge_polydata(polydata_list: Sequence[vtkPolyData | vtkAppendPolyData]) -> vtkAppendPolyData:
    """Merges a list of polydata into a vtkAppendPolyData

    Args:
        polydata_list (list[vtkPolyData]): List of vtkPolyData

    Returns:
        vtkAppendPolyData: Appended polydata object
    """
    append_poly_data = vtkAppendPolyData()

    for polydata in polydata_list:
        append_poly_data.AddInputData(polydata)

    append_poly_data.Update()
    return append_poly_data


def write_polydata(output_filename: pathlib.Path, poly_data: vtkPolyData | vtkAppendPolyData):
    """Writes a polydata object to file output

    Args:
        output_filename (pathlib.Path): Output file path
        poly_data (vtkPolyData | vtkAppendPolyData): Polydata object
    """
    writer = vtkXMLPolyDataWriter()
    create_folders_for_file(output_filename)
    writer.SetFileName(output_filename.as_posix())
    if isinstance(poly_data, vtkPolyData):
        writer.SetInputData(poly_data)
    else:
        writer.SetInputData(poly_data.GetOutput())
    writer.SetDataModeToAscii()
    writer.Write()
