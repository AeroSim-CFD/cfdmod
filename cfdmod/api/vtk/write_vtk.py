# Backward compatibility shim. Import from cfdmod.io.vtk.write_vtk instead.
from cfdmod.io.vtk.write_vtk import (
    create_polydata_for_cell_data,
    merge_polydata,
    read_polydata,
    write_polydata,
    drop_all_scalars_except,
    envelope_vtks,
)

__all__ = [
    "create_polydata_for_cell_data",
    "merge_polydata",
    "read_polydata",
    "write_polydata",
    "drop_all_scalars_except",
    "envelope_vtks",
]
