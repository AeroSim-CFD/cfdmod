# Backward compatibility shim. Import from cfdmod.io.geometry instead.
__all__ = ["read_stl", "export_stl"]

from cfdmod.io.geometry.STL import read_stl, export_stl
