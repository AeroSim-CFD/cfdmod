# Backward compatibility shim. Import from cfdmod.io.vtk.probe_vtm instead.
from cfdmod.io.vtk.probe_vtm import (
    read_vtm,
    create_line,
    probe_over_line,
    get_array_from_filter,
)

__all__ = ["read_vtm", "create_line", "probe_over_line", "get_array_from_filter"]
