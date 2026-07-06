"""Multi-format mesh loader.

The v3 pressure pipeline ops (``mesh_attach``, ``body_grouping``) accept
the mesh as a file path resolved against the template's root. This
helper handles the path dispatch by suffix:

- ``.lnas`` -> :func:`lnas.LnasFormat.from_file` (preserves authored surfaces)
- ``.stl``  -> :func:`lnas.LnasFormat.from_stl` (single ``"all"`` surface)
- ``.h5``   -> read ``/Triangles + /Geometry``, single ``"all"`` surface
- ``.xdmf`` -> redirect to the sibling ``.h5`` (XDMF is metadata only)

Files that already have authored surface labels (``.lnas``) keep them.
The other formats produce one synthetic ``"all"`` surface covering
every triangle, so YAML templates targeting that surface (or
``surfaces: []`` for "use everything") work uniformly.
"""

from __future__ import annotations

__all__ = ["load_mesh", "mesh_from_h5"]

import pathlib

import h5py
import numpy as np
from lnas import LnasFormat, LnasGeometry


def _resolve_lnas_version() -> str:
    """Pick a version string the installed lnas reader will accept.

    lnas exposes ``_CURRENT_VERSION`` (e.g. ``"v0.5.2"``) on
    :mod:`lnas.fmt`; we use that when available so cfdmod tracks the
    library it's pinned against. If the symbol moves or disappears, fall
    back to a hard-coded value compatible with lnas's major-version check.
    """
    try:
        from lnas import fmt as _lnas_fmt

        version = getattr(_lnas_fmt, "_CURRENT_VERSION", None)
        if isinstance(version, str) and version:
            return version
    except Exception:
        pass
    return "v0.5.2"


_LNAS_VERSION = _resolve_lnas_version()
_DEFAULT_SURFACE_NAME = "all"


def mesh_from_h5(h5_path: pathlib.Path) -> LnasFormat:
    """Build an :class:`LnasFormat` from an HDF5 file's ``/Triangles +
    /Geometry`` datasets, with one synthetic surface covering every triangle.

    Used both by :func:`load_mesh` (for ``.h5``/``.xdmf`` inputs) and as the
    fallback when a pipeline call omits ``mesh_path`` and the geometry has to
    come straight from the body or cp timeseries file.
    """
    with h5py.File(h5_path, "r") as f:
        if "Triangles" not in f or "Geometry" not in f:
            raise ValueError(
                f"{h5_path} has no /Triangles + /Geometry; cannot build a mesh from it."
            )
        triangles = f["Triangles"][:].astype(np.int32)
        vertices = f["Geometry"][:].astype(np.float64)

    geom = LnasGeometry(vertices=vertices, triangles=triangles)
    surfaces = {_DEFAULT_SURFACE_NAME: np.arange(len(triangles), dtype=np.int32)}
    return LnasFormat(version=_LNAS_VERSION, geometry=geom, surfaces=surfaces)


def load_mesh(source: pathlib.Path | LnasFormat) -> LnasFormat:
    """Resolve a mesh from any supported source.

    Args:
        source: Either an :class:`LnasFormat` (returned as-is) or a path
            ending in ``.lnas``, ``.stl``, ``.h5``, or ``.xdmf``.

    Returns:
        LnasFormat ready to drive the pressure pipeline.
    """
    if isinstance(source, LnasFormat):
        return source

    path = pathlib.Path(source)
    suffix = path.suffix.lower()

    if suffix == ".lnas":
        return LnasFormat.from_file(path)

    if suffix == ".stl":
        return LnasFormat.from_stl(path)

    if suffix == ".xdmf":
        h5_sibling = path.with_suffix(".h5")
        if not h5_sibling.exists():
            raise FileNotFoundError(
                f"{path} references mesh data but the sibling H5 {h5_sibling} is missing."
            )
        return mesh_from_h5(h5_sibling)

    if suffix == ".h5":
        return mesh_from_h5(path)

    raise ValueError(
        f"Unsupported mesh format {suffix!r} for {path}. Expected one of: .lnas, .stl, .h5, .xdmf."
    )
