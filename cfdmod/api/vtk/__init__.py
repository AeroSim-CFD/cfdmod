# Backward compatibility shim. Import from cfdmod.io.vtk instead.
__all__ = [
    "read_vtm",
    "create_line",
    "probe_over_line",
    "get_array_from_filter",
    "create_polydata_for_cell_data",
    "write_polydata",
    "read_polydata",
    "merge_polydata",
]

from cfdmod.io.vtk.probe_vtm import (
    read_vtm,
    create_line,
    probe_over_line,
    get_array_from_filter,
)
from cfdmod.io.vtk.write_vtk import (
    create_polydata_for_cell_data,
    write_polydata,
    read_polydata,
    merge_polydata,
)
