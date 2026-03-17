"""Notebook convenience utilities.

Provides lightweight helpers for exploratory work in Jupyter notebooks.
"""
import pathlib
import pprint
from typing import Union

from lnas import LnasFormat

from cfdmod.config.hashable import HashableConfig
from cfdmod.io.geometry.STL import read_stl


def mesh_summary(path: pathlib.Path) -> None:
    """Print a summary of an LNAS or STL mesh file.

    Args:
        path: Path to .lnas or .stl file.
    """
    path = pathlib.Path(path)
    suffix = path.suffix.lower()
    if suffix == ".lnas":
        fmt = LnasFormat.from_file(path)
        geom = fmt.geometry
        verts = geom.vertices
        tris = geom.triangles
    elif suffix == ".stl":
        triangles, normals = read_stl(path)
        n_tris = len(triangles)
        verts_flat = triangles.reshape(-1, 3)
        mins = verts_flat.min(axis=0)
        maxs = verts_flat.max(axis=0)
        print(f"STL mesh: {path.name}")
        print(f"  triangles : {n_tris}")
        print(f"  vertices  : {n_tris * 3} (flat, may have duplicates)")
        print(f"  x bounds  : [{mins[0]:.3f}, {maxs[0]:.3f}]")
        print(f"  y bounds  : [{mins[1]:.3f}, {maxs[1]:.3f}]")
        print(f"  z bounds  : [{mins[2]:.3f}, {maxs[2]:.3f}]")
        return
    else:
        raise ValueError(f"Unsupported mesh format: {suffix}. Use .lnas or .stl")

    mins = verts.min(axis=0)
    maxs = verts.max(axis=0)
    print(f"LNAS mesh: {path.name}")
    print(f"  triangles : {len(tris)}")
    print(f"  vertices  : {len(verts)}")
    print(f"  x bounds  : [{mins[0]:.3f}, {maxs[0]:.3f}]")
    print(f"  y bounds  : [{mins[1]:.3f}, {maxs[1]:.3f}]")
    print(f"  z bounds  : [{mins[2]:.3f}, {maxs[2]:.3f}]")


def show_config(config: HashableConfig) -> None:
    """Pretty-print a HashableConfig as a dictionary.

    Args:
        config: Any HashableConfig instance (LoftCaseConfig, CpCaseConfig, etc.)
    """
    pprint.pprint(config.to_dict())


def load_lnas(path: Union[str, pathlib.Path]) -> LnasFormat:
    """Load an LNAS file and return the LnasFormat object.

    Args:
        path: Path to .lnas file.

    Returns:
        LnasFormat object ready for use.
    """
    return LnasFormat.from_file(pathlib.Path(path))
