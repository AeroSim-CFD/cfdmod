"""XDMF+H5 reader/writer for timeseries and stats.

Used by the pressure module for both per-timestep timeseries and combined stats output.
"""

from __future__ import annotations

__all__ = [
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
]

import pathlib
import xml.etree.ElementTree as ET
from xml.dom import minidom

import h5py
import numpy as np


def get_pressure_keys(
    h5_path: pathlib.Path, group: str = "pressure"
) -> list[tuple[float, str]]:
    """Return sorted (float_time, key_str) pairs from H5 group.

    Keys are expected in the form t{T} where T is the float time value.
    """
    with h5py.File(h5_path, "r") as f:
        keys = list(f[group].keys())
    result = [(float(k[1:]), k) for k in keys]
    return sorted(result, key=lambda x: x[0])


def filter_keys_by_range(
    keys: list[tuple[float, str]], timestep_range: tuple[float, float]
) -> list[tuple[float, str]]:
    """Filter keys to [t_min, t_max] inclusive."""
    t_min, t_max = timestep_range
    return [(t, k) for t, k in keys if t_min <= t <= t_max]


def read_step(h5_path: pathlib.Path, key: str, group: str) -> np.ndarray:
    """Read a single timestep array from an H5 group."""
    with h5py.File(h5_path, "r") as f:
        return f[group][key][:]


def read_timeseries_meta(h5_path: pathlib.Path) -> dict:
    """Read /meta group.

    Returns dict with keys:
        time_steps: float64 array of raw simulation time values
        time_normalized: float64 array of normalized time values
        region_labels: list[str] (only if present in file)
    """
    with h5py.File(h5_path, "r") as f:
        meta = f["meta"]
        result: dict = {
            "time_steps": meta["time_steps"][:],
            "time_normalized": meta["time_normalized"][:],
        }
        if "region_labels" in meta:
            result["region_labels"] = [s.decode() for s in meta["region_labels"][:]]
    return result


def write_timeseries_step(
    h5_path: pathlib.Path,
    group: str,
    key: str,
    data: np.ndarray,
    mode: str = "a",
) -> None:
    """Append a single timestep array to an H5 group."""
    with h5py.File(h5_path, mode) as f:
        grp = f.require_group(group)
        if key in grp:
            del grp[key]
        grp.create_dataset(key, data=data.astype(np.float64))


def write_timeseries_meta(
    h5_path: pathlib.Path,
    time_steps: np.ndarray,
    time_normalized: np.ndarray,
    region_labels: list[str] | None = None,
) -> None:
    """Write /meta datasets (time arrays + optional region labels)."""
    with h5py.File(h5_path, "a") as f:
        meta = f.require_group("meta")
        for key in ("time_steps", "time_normalized", "region_labels"):
            if key in meta:
                del meta[key]
        meta.create_dataset("time_steps", data=np.array(time_steps, dtype=np.float64))
        meta.create_dataset(
            "time_normalized", data=np.array(time_normalized, dtype=np.float64)
        )
        if region_labels is not None:
            encoded = [s.encode() for s in region_labels]
            meta.create_dataset("region_labels", data=np.array(encoded))


def write_timeseries_geometry(
    h5_path: pathlib.Path,
    triangles: np.ndarray,
    vertices: np.ndarray,
) -> None:
    """Write /Triangles and /Geometry to H5 file (only needed once per file)."""
    with h5py.File(h5_path, "a") as f:
        for key in ("Triangles", "Geometry"):
            if key in f:
                del f[key]
        f.create_dataset("Triangles", data=triangles.astype(np.int32))
        f.create_dataset("Geometry", data=vertices.astype(np.float64))


def write_temporal_xdmf(
    h5_path: pathlib.Path,
    xdmf_path: pathlib.Path,
    group: str,
) -> None:
    """Write temporal XDMF XML for a timeseries H5.

    Reads /Triangles, /Geometry, and /{group}/t{T} keys from h5_path.
    Produces a temporal collection (one grid per timestep) compatible with ParaView.
    """
    with h5py.File(h5_path, "r") as f:
        n_tri = f["Triangles"].shape[0]
        n_verts = f["Geometry"].shape[0]
        keys = sorted(f[group].keys(), key=lambda k: float(k[1:]))

    h5_name = h5_path.name
    root = ET.Element("Xdmf", Version="3.0")
    domain = ET.SubElement(root, "Domain")
    collection = ET.SubElement(
        domain,
        "Grid",
        Name="TimeSeries",
        GridType="Collection",
        CollectionType="Temporal",
    )

    for key in keys:
        t_val = float(key[1:])
        grid = ET.SubElement(collection, "Grid", Name=key, GridType="Uniform")
        ET.SubElement(grid, "Time", Value=str(t_val))

        topo = ET.SubElement(
            grid,
            "Topology",
            TopologyType="Triangle",
            NumberOfElements=str(n_tri),
        )
        topo_item = ET.SubElement(
            topo,
            "DataItem",
            Format="HDF",
            DataType="Int",
            Dimensions=f"{n_tri} 3",
        )
        topo_item.text = f"{h5_name}:/Triangles"

        geom = ET.SubElement(grid, "Geometry", GeometryType="XYZ")
        geom_item = ET.SubElement(
            geom,
            "DataItem",
            Format="HDF",
            DataType="Float",
            Precision="8",
            Dimensions=f"{n_verts} 3",
        )
        geom_item.text = f"{h5_name}:/Geometry"

        attr = ET.SubElement(grid, "Attribute", Name=group, Center="Cell")
        attr_item = ET.SubElement(
            attr,
            "DataItem",
            Format="HDF",
            DataType="Float",
            Precision="8",
            Dimensions=str(n_tri),
        )
        attr_item.text = f"{h5_name}:/{group}/{key}"

    _write_pretty_xml(root, xdmf_path)


def write_stats_field(
    h5_path: pathlib.Path,
    group: str,
    stat_name: str,
    values: np.ndarray,
    triangles: np.ndarray | None = None,
    vertices: np.ndarray | None = None,
) -> None:
    """Write a single stats field (e.g. /cp/mean) to the combined H5.

    If triangles/vertices are provided and not yet in the file, writes
    /Triangles and /Geometry.
    """
    with h5py.File(h5_path, "a") as f:
        if "Triangles" not in f and triangles is not None:
            f.create_dataset("Triangles", data=triangles.astype(np.int32))
        if "Geometry" not in f and vertices is not None:
            f.create_dataset("Geometry", data=vertices.astype(np.float64))
        grp = f.require_group(group)
        if stat_name in grp:
            del grp[stat_name]
        grp.create_dataset(stat_name, data=values.astype(np.float64))


def write_stats_xdmf(h5_path: pathlib.Path, xdmf_path: pathlib.Path) -> None:
    """(Re)generate the static XDMF XML for the combined stats H5.

    Scans all /{group}/{stat} datasets and lists them as Attributes on the
    mesh grid. Called after each processing step to keep XDMF up to date.
    """
    _SKIP = {"Triangles", "Geometry", "meta"}

    with h5py.File(h5_path, "r") as f:
        n_tri = f["Triangles"].shape[0]
        n_verts = f["Geometry"].shape[0]
        fields: list[tuple[str, str]] = []
        for grp_name in f.keys():
            if grp_name in _SKIP:
                continue
            grp = f[grp_name]
            if isinstance(grp, h5py.Group):
                for stat_name in grp.keys():
                    fields.append((grp_name, stat_name))

    h5_name = h5_path.name
    root = ET.Element("Xdmf", Version="3.0")
    domain = ET.SubElement(root, "Domain")
    grid = ET.SubElement(domain, "Grid", Name="Results", GridType="Uniform")

    topo = ET.SubElement(
        grid,
        "Topology",
        TopologyType="Triangle",
        NumberOfElements=str(n_tri),
    )
    topo_item = ET.SubElement(
        topo,
        "DataItem",
        Format="HDF",
        DataType="Int",
        Dimensions=f"{n_tri} 3",
    )
    topo_item.text = f"{h5_name}:/Triangles"

    geom = ET.SubElement(grid, "Geometry", GeometryType="XYZ")
    geom_item = ET.SubElement(
        geom,
        "DataItem",
        Format="HDF",
        DataType="Float",
        Precision="8",
        Dimensions=f"{n_verts} 3",
    )
    geom_item.text = f"{h5_name}:/Geometry"

    for grp_name, stat_name in sorted(fields):
        attr = ET.SubElement(
            grid,
            "Attribute",
            Name=f"{grp_name}/{stat_name}",
            Center="Cell",
        )
        attr_item = ET.SubElement(
            attr,
            "DataItem",
            Format="HDF",
            DataType="Float",
            Precision="8",
            Dimensions=str(n_tri),
        )
        attr_item.text = f"{h5_name}:/{grp_name}/{stat_name}"

    _write_pretty_xml(root, xdmf_path)


def _write_pretty_xml(root: ET.Element, path: pathlib.Path) -> None:
    """Write XML element to file with indentation."""
    raw = ET.tostring(root, encoding="unicode")
    pretty = minidom.parseString(raw).toprettyxml(indent="  ")
    lines = pretty.split("\n")
    content = "\n".join(lines[1:])
    with open(path, "w") as fh:
        fh.write('<?xml version="1.0" ?>\n')
        fh.write('<!DOCTYPE Xdmf SYSTEM "Xdmf.dtd">\n')
        fh.write(content)
