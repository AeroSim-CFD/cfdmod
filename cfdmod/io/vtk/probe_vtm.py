__all__ = ["read_vtm", "create_line", "probe_over_line", "get_array_from_filter"]

import pathlib
from typing import Sequence

import numpy as np
from vtk import vtkCompositeDataProbeFilter, vtkLineSource, vtkXMLMultiBlockDataReader
from vtk.util.numpy_support import vtk_to_numpy


def read_vtm(multiblock_file: pathlib.Path) -> vtkXMLMultiBlockDataReader:
    reader = vtkXMLMultiBlockDataReader()
    reader.SetFileName(multiblock_file.as_posix())
    reader.Update()
    return reader


def create_line(p1: Sequence[float], p2: Sequence[float], numPoints: int) -> vtkLineSource:
    """Create the line along which you want to sample"""
    line = vtkLineSource()
    line.SetResolution(numPoints)
    line.SetPoint1(p1)
    line.SetPoint2(p2)
    line.Update()
    return line


def probe_over_line(
    line: vtkLineSource, reader: vtkXMLMultiBlockDataReader
) -> vtkCompositeDataProbeFilter:
    """Interpolate the data from the VTK-file on the created line"""
    probe = vtkCompositeDataProbeFilter()  # For multiblock datasets only
    probe.SetInputData(line.GetOutput())
    probe.SetSourceData(reader.GetOutput())
    probe.Update()

    return probe


def get_array_from_filter(probe_filter: vtkCompositeDataProbeFilter, array_lbl: str) -> np.ndarray:
    probed_data = vtk_to_numpy(probe_filter.GetOutput().GetPointData().GetArray(array_lbl))
    return probed_data
