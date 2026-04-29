"""XDMF+H5 reader/writer for timeseries and stats.

Two layout conventions are used:

- **Timeseries files** (e.g. ``cp.time_series.h5``, body Cf timeseries) embed a
  single mesh at the file root: ``/Triangles`` and ``/Geometry``. Per-timestep
  arrays live under one or more groups (``/cp/t{T}``, ``/cf_x/t{T}``, ...).
  ``write_temporal_xdmf`` reads the root mesh and emits one Grid per timestep
  with one Attribute per group.

- **Stats results files** (``results.h5``) embed a separate mesh inside *each*
  leaf group, alongside the per-stat datasets:
  ``/{path}/{Triangles, Geometry, mean, rms, ...}``. ``write_stats_xdmf`` walks
  the tree and emits one Grid per group that has both Triangles and Geometry.
  This lets a single file describe stats over different sub-meshes (e.g. a
  sliced regions mesh for Ce, body subsets for Cf/Cm) without length collisions.
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
    "write_processing_metadata",
    "read_processing_metadata",
]

import datetime as _dt
import pathlib
import xml.etree.ElementTree as ET
from xml.dom import minidom

import h5py
import numpy as np
from ruamel.yaml import YAML


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
    group: str | list[str],
) -> None:
    """Write temporal XDMF XML for a timeseries H5.

    Reads ``/Triangles``, ``/Geometry``, and ``/{group}/t{T}`` keys from
    ``h5_path``. Produces a temporal collection (one Grid per timestep)
    compatible with ParaView. When multiple groups are supplied, each Grid
    carries one Attribute per group (e.g. for Cf with x/y/z directions).

    Args:
        h5_path: Source H5 file.
        xdmf_path: Output .xdmf path.
        group: Single group name or list of group names. The first group's
            keys define the time axis.
    """
    groups = [group] if isinstance(group, str) else list(group)
    if not groups:
        raise ValueError("At least one group is required")

    with h5py.File(h5_path, "r") as f:
        n_tri = f["Triangles"].shape[0]
        n_verts = f["Geometry"].shape[0]
        keys = sorted(f[groups[0]].keys(), key=lambda k: float(k[1:]))

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

        for grp_name in groups:
            attr = ET.SubElement(grid, "Attribute", Name=grp_name, Center="Cell")
            attr_item = ET.SubElement(
                attr,
                "DataItem",
                Format="HDF",
                DataType="Float",
                Precision="8",
                Dimensions=str(n_tri),
            )
            attr_item.text = f"{h5_name}:/{grp_name}/{key}"

    _write_pretty_xml(root, xdmf_path)


def write_stats_field(
    h5_path: pathlib.Path,
    group: str,
    stat_name: str,
    values: np.ndarray,
    triangles: np.ndarray | None = None,
    vertices: np.ndarray | None = None,
) -> None:
    """Write a stat dataset to ``<group>/<stat_name>``.

    When ``triangles`` and ``vertices`` are provided, ``<group>/Triangles`` and
    ``<group>/Geometry`` are created if not already present. Each leaf group
    therefore carries its own embedded mesh; ``write_stats_xdmf`` discovers
    these and emits one Grid per group.
    """
    with h5py.File(h5_path, "a") as f:
        grp = f.require_group(group)
        if triangles is not None and "Triangles" not in grp:
            grp.create_dataset("Triangles", data=triangles.astype(np.int32))
        if vertices is not None and "Geometry" not in grp:
            grp.create_dataset("Geometry", data=vertices.astype(np.float64))
        if stat_name in grp:
            del grp[stat_name]
        grp.create_dataset(stat_name, data=values.astype(np.float64))


def write_stats_xdmf(h5_path: pathlib.Path, xdmf_path: pathlib.Path) -> None:
    """(Re)generate the static XDMF XML for a stats H5.

    Walks every group in ``h5_path``; emits one Grid per group that contains
    both ``Triangles`` and ``Geometry``. Sibling datasets in such a group
    (other than Triangles/Geometry) become Cell Attributes on that Grid.
    """
    grids: list[tuple[str, int, int, list[str]]] = []
    with h5py.File(h5_path, "r") as f:

        def visitor(name: str, obj) -> None:
            if not isinstance(obj, h5py.Group):
                return
            if "Triangles" not in obj or "Geometry" not in obj:
                return
            n_tri = obj["Triangles"].shape[0]
            n_verts = obj["Geometry"].shape[0]
            stats = sorted(
                k
                for k in obj.keys()
                if k not in ("Triangles", "Geometry")
                and isinstance(obj[k], h5py.Dataset)
            )
            grids.append((name, n_tri, n_verts, stats))

        f.visititems(visitor)

    if not grids:
        raise ValueError(
            f"{h5_path} has no group with both Triangles and Geometry to emit XDMF for."
        )

    h5_name = h5_path.name
    root = ET.Element("Xdmf", Version="3.0")
    domain = ET.SubElement(root, "Domain")

    for grp_path, n_tri, n_verts, stats in sorted(grids):
        grid = ET.SubElement(domain, "Grid", Name=grp_path, GridType="Uniform")

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
        topo_item.text = f"{h5_name}:/{grp_path}/Triangles"

        geom = ET.SubElement(grid, "Geometry", GeometryType="XYZ")
        geom_item = ET.SubElement(
            geom,
            "DataItem",
            Format="HDF",
            DataType="Float",
            Precision="8",
            Dimensions=f"{n_verts} 3",
        )
        geom_item.text = f"{h5_name}:/{grp_path}/Geometry"

        for stat_name in stats:
            attr = ET.SubElement(
                grid,
                "Attribute",
                Name=f"{grp_path}/{stat_name}",
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
            attr_item.text = f"{h5_name}:/{grp_path}/{stat_name}"

    _write_pretty_xml(root, xdmf_path)


def write_processing_metadata(
    h5_path: pathlib.Path,
    group: str,
    config: dict,
    *,
    extra: dict | None = None,
) -> None:
    """Embed the post-processing parameters used to produce ``group`` as
    HDF5 attributes on that group, plus the YAML serialization as a
    sibling string dataset for round-trip reproducibility.

    Attributes:
        config_yaml: full config serialized to YAML (string)
        produced_at: ISO-8601 UTC timestamp
        cfdmod_version: package version
        plus every key from ``extra`` (e.g. {'body_h5': '...', 'probe_h5': '...'})

    Stored under ``{group}/processing_metadata/config.yaml`` (string dataset)
    so it remains human-inspectable via ``h5dump`` even when attribute
    inspection isn't available.
    """
    from io import StringIO

    yaml = YAML(typ="safe")
    yaml.default_flow_style = False
    buf = StringIO()
    yaml.dump(config, buf)
    yaml_text = buf.getvalue()

    try:
        from importlib.metadata import version as _pkg_version

        pkg_version = _pkg_version("aerosim-cfdmod")
    except Exception:
        pkg_version = "unknown"

    with h5py.File(h5_path, "a") as f:
        grp = f.require_group(group)
        meta = grp.require_group("processing_metadata")
        for key in ("config.yaml", "config_yaml"):
            if key in meta:
                del meta[key]
        meta.create_dataset("config.yaml", data=yaml_text)
        meta.attrs["produced_at"] = _dt.datetime.now(_dt.timezone.utc).isoformat()
        meta.attrs["cfdmod_version"] = pkg_version
        if extra:
            for k, v in extra.items():
                meta.attrs[k] = str(v)


def read_processing_metadata(h5_path: pathlib.Path, group: str) -> dict:
    """Read the metadata written by :func:`write_processing_metadata`.

    Returns dict with keys ``config`` (parsed YAML), ``produced_at``,
    ``cfdmod_version``, and any ``extra`` keys recorded at write time.
    """
    yaml = YAML(typ="safe")
    with h5py.File(h5_path, "r") as f:
        meta = f[group]["processing_metadata"]
        text = meta["config.yaml"][()]
        if isinstance(text, bytes):
            text = text.decode()
        result: dict = {"config": yaml.load(text)}
        for k, v in meta.attrs.items():
            result[k] = v.decode() if isinstance(v, bytes) else v
    return result


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
