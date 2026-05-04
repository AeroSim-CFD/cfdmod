__all__ = [
    "read_stl",
    "export_stl",
    "TransformationConfig",
    "create_regions_mesh",
    # vtk-backed (lazy: require the `vtk` extras to be installed)
    "read_vtm",
    "create_line",
    "probe_over_line",
    "get_array_from_filter",
    "create_polydata_for_cell_data",
    "write_polydata",
    "read_polydata",
    "merge_polydata",
    # XDMF+H5
    "get_pressure_keys",
    "filter_keys_by_range",
    "read_step",
    "read_timeseries_meta",
    "write_timeseries_step",
    "write_timeseries_meta",
    "write_timeseries_geometry",
    "write_temporal_xdmf",
    "write_stats_field",
    "write_stats_xdmf",
    "write_processing_metadata",
    "read_processing_metadata",
    # mesh resolver
    "load_mesh",
    "mesh_from_h5",
    # timeseries DataFrame helpers
    "read_timeseries_df",
    "to_csv",
    "plot_timeseries",
    # inspect helpers (debug)
    "inspect_h5",
    "read_all_timesteps",
]

from cfdmod.io.geometry.STL import read_stl, export_stl
from cfdmod.io.geometry.transformation_config import TransformationConfig
from cfdmod.io.geometry.region_meshing import create_regions_mesh
from cfdmod.io.inspect import inspect_h5, read_all_timesteps
from cfdmod.io.mesh import load_mesh, mesh_from_h5
from cfdmod.io.timeseries import plot_timeseries, read_timeseries_df, to_csv
from cfdmod.io.xdmf import (
    get_pressure_keys,
    filter_keys_by_range,
    read_processing_metadata,
    read_step,
    read_timeseries_meta,
    write_processing_metadata,
    write_timeseries_step,
    write_timeseries_meta,
    write_timeseries_geometry,
    write_temporal_xdmf,
    write_stats_field,
    write_stats_xdmf,
)

# vtk-backed helpers are lazy-loaded so `import cfdmod` does not require the
# optional `vtk` extras. Install with `pip install aerosim-cfdmod[vtk]` to
# enable. Accessing the names below triggers the import on first use.
_LAZY_VTK = {
    "read_vtm": ("cfdmod.io.vtk.probe_vtm", "read_vtm"),
    "create_line": ("cfdmod.io.vtk.probe_vtm", "create_line"),
    "probe_over_line": ("cfdmod.io.vtk.probe_vtm", "probe_over_line"),
    "get_array_from_filter": ("cfdmod.io.vtk.probe_vtm", "get_array_from_filter"),
    "create_polydata_for_cell_data": (
        "cfdmod.io.vtk.write_vtk",
        "create_polydata_for_cell_data",
    ),
    "write_polydata": ("cfdmod.io.vtk.write_vtk", "write_polydata"),
    "read_polydata": ("cfdmod.io.vtk.write_vtk", "read_polydata"),
    "merge_polydata": ("cfdmod.io.vtk.write_vtk", "merge_polydata"),
}


def __getattr__(name):
    if name in _LAZY_VTK:
        import importlib

        module_path, attr = _LAZY_VTK[name]
        try:
            module = importlib.import_module(module_path)
        except ImportError as exc:
            raise ImportError(
                f"cfdmod.io.{name} requires the optional 'vtk' extras. "
                "Install with: pip install aerosim-cfdmod[vtk]"
            ) from exc
        value = getattr(module, attr)
        globals()[name] = value
        return value
    raise AttributeError(f"module 'cfdmod.io' has no attribute {name!r}")
