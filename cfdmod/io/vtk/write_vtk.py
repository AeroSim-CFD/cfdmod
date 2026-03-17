import pathlib
from typing import Literal, Sequence

import pandas as pd
import vtk
from lnas import LnasGeometry

from cfdmod.utils import create_folders_for_file


def _mkVtkIdList(it) -> vtk.vtkIdList:
    """Makes a vtkIdList from a Python iterable. I'm kinda surprised that
     this is necessary, since I assumed that this kind of thing would
     have been built into the wrapper and happen transparently, but it
     seems not.

    Args:
        it (Iterable): A python iterable.

    Returns:
        vtk.vtkIdList: A vtkIdList
    """
    vil = vtk.vtkIdList()
    for i in it:
        vil.InsertNextId(int(i))
    return vil


def create_polydata_for_cell_data(data: pd.DataFrame, mesh: LnasGeometry) -> vtk.vtkPolyData:
    """Creates a vtk.vtkPolyData for cell data combined with mesh description

    Args:
        data (pd.DataFrame): Compiled cell data. It supports table and matrix data formats.
            In matrix form, each column represents a point, and each row identifies the scalar label.
            In table form, there is a column with point indexes, and other columns for scalar data.
        mesh (LnasGeometry): Mesh description

    Returns:
        vtk.vtkPolyData: Extracted polydata
    """
    # We'll create the building blocks of polydata including data attributes.
    polyData = vtk.vtkPolyData()
    points = vtk.vtkPoints()
    polys = vtk.vtkCellArray()

    # Load the point, cell, and data attributes.
    for i, xi in enumerate(mesh.vertices):
        points.InsertPoint(i, xi)
    for pt in mesh.triangles:
        polys.InsertNextCell(_mkVtkIdList(pt))

    # We now assign the pieces to the vtkPolyData.
    polyData.SetPoints(points)
    polyData.SetPolys(polys)

    scalars_lbls, point_idx = None, None
    if "point_idx" in data.columns:
        # Table form dataframe
        scalars_lbls = [c for c in data.columns if c != "point_idx"]
        point_idx = data["point_idx"].to_numpy()
    else:
        # Matrix form dataframe
        scalars_lbls = data["scalar"]
        point_idx = [int(c) for c in data.columns if c != "scalar"]
    for scalar_index, scalar_lbl in enumerate(scalars_lbls):
        scalars = vtk.vtkFloatArray()
        scalars.SetName(scalar_lbl)
        scalar_data = None

        if "point_idx" in data.columns:
            # Table form dataframe, scalar is in columns
            scalar_data = data[scalar_lbl].to_numpy()
        else:
            # Matrix form dataframe, scalar is in rows
            scalar_data = data.iloc[scalar_index][
                [col for col in data.columns if col != "scalar"]
            ].to_numpy()
        for i, value in zip(point_idx, scalar_data):
            scalars.InsertTuple1(i, value)
        polyData.GetCellData().AddArray(scalars)

    return polyData


def merge_polydata(
    polydata_list: Sequence[vtk.vtkPolyData | vtk.vtkAppendPolyData],
) -> vtk.vtkAppendPolyData:
    """Merges a list of polydata into a vtkAppendPolyData

    Args:
        polydata_list (Sequence[vtk.vtkPolyData | vtk.vtkAppendPolyData]): List of vtkPolyData

    Returns:
        vtk.vtkAppendPolyData: Appended polydata object
    """
    append_poly_data = vtk.vtkAppendPolyData()

    for polydata in polydata_list:
        append_poly_data.AddInputData(polydata)

    append_poly_data.Update()
    return append_poly_data


def read_polydata(file_path: pathlib.Path) -> vtk.vtkPolyData:
    """Reads polydata from file

    Args:
        file_path (pathlib.Path): File path

    Returns:
        vtkPolyData: Read polydata
    """
    reader = vtk.vtkXMLPolyDataReader()
    reader.SetFileName(file_path)
    reader.Update()

    polydata = reader.GetOutput()

    return polydata


def write_polydata(
    output_filename: pathlib.Path, poly_data: vtk.vtkPolyData | vtk.vtkAppendPolyData
):
    """Writes a polydata object to file output

    Args:
        output_filename (pathlib.Path): Output file path
        poly_data (vtk.vtkPolyData | vtk.vtkAppendPolyData): Polydata object
    """
    writer = vtk.vtkXMLPolyDataWriter()
    create_folders_for_file(output_filename)
    writer.SetFileName(output_filename.as_posix())
    if isinstance(poly_data, vtk.vtkPolyData):
        writer.SetInputData(poly_data)
    else:
        writer.SetInputData(poly_data.GetOutput())
    writer.SetDataModeToAscii()
    writer.Write()


def drop_all_scalars_except(polydata: vtk.vtkPolyData, scalar: str) -> None:
    """Removes all cell scalar arrays from polydata except the one named `scalar`."""

    cell_data = polydata.GetCellData()
    names_to_remove = [
        cell_data.GetArrayName(i)
        for i in range(cell_data.GetNumberOfArrays())
        if cell_data.GetArrayName(i) != scalar
    ]

    for name in names_to_remove:
        cell_data.RemoveArray(name)


def envelope_vtks(
    polydatas: list[vtk.vtkPolyData], scalar: str, stats: Literal["min", "max"]
) -> vtk.vtkPolyData:
    if stats not in {"min", "max"}:
        raise ValueError("stats must be 'min' or 'max'")
    if len(polydatas) == 0:
        raise ValueError("Empty input list")

    # Start with a deep copy of the first polydata
    envelope_polydata = vtk.vtkPolyData()
    envelope_polydata.DeepCopy(polydatas[0])

    # Extract the scalar array from the envelope copy (this will be updated)
    envelope_array = envelope_polydata.GetCellData().GetArray(scalar)
    n_cells = envelope_polydata.GetNumberOfCells()

    # Loop through all other polydata
    for polydata in polydatas[1:]:
        other_array = polydata.GetCellData().GetArray(scalar)
        for i in range(n_cells):
            current_val = envelope_array.GetValue(i)
            new_val = other_array.GetValue(i)
            if stats == "max":
                envelope_array.SetValue(i, max(current_val, new_val))
            elif stats == "min":
                envelope_array.SetValue(i, min(current_val, new_val))

    envelope_array.Modified()
    return envelope_polydata
