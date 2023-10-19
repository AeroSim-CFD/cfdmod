import pathlib

import pandas as pd
from nassu.lnas import LagrangianGeometry
from vtk import (
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


def create_polydata_for_cell_data(data: pd.DataFrame, mesh: LagrangianGeometry) -> vtkPolyData:
    """Creates a vtkPolyData for cell data combined with mesh description

    Args:
        data (pd.DataFrame): Compiled cell data
        mesh (LagrangianGeometry): Mesh description

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


def write_polydata(output_filename: pathlib.Path, poly_data: vtkPolyData):
    """Writes a polydata object to file output

    Args:
        output_filename (pathlib.Path): Output file path
        poly_data (vtkPolyData): Polydata object
    """
    writer = vtkXMLPolyDataWriter()
    create_folders_for_file(output_filename)
    writer.SetFileName(output_filename.as_posix())
    writer.SetInputData(poly_data)
    writer.SetDataModeToAscii()
    writer.Write()
