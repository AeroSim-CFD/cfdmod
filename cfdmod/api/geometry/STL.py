# Backward compatibility shim. Import from cfdmod.io.geometry.STL instead.
from cfdmod.io.geometry.STL import read_stl, export_stl

__all__ = ["read_stl", "export_stl"]
