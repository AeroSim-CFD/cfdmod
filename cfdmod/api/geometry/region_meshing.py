# Backward compatibility shim. Import from cfdmod.io.geometry.region_meshing instead.
from cfdmod.io.geometry.region_meshing import (
    triangulate_tri,
    slice_triangle,
    slice_surface,
    create_regions_mesh,
)

__all__ = ["triangulate_tri", "slice_triangle", "slice_surface", "create_regions_mesh"]
