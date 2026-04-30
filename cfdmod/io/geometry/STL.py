import pathlib
from io import BytesIO

import numpy as np

from cfdmod.utils import create_folders_for_file

__all__ = ["export_stl", "read_stl"]


def read_stl(filename: pathlib.Path) -> tuple[np.ndarray, np.ndarray]:
    """Read buffer content as STL file

    Args:
        buff (io.BufferedReader): buffer to read from

    Returns:
        tuple[np.ndarray, np.ndarray]: return STL representation as (triangles, normals).
    """

    # pass header
    with open(filename, "rb") as f:
        buff = BytesIO(f.read())

    buff.read(80)  # Header
    # Read number of triangles

    n_triangles = np.frombuffer(buff.read(4), dtype=np.uint32)[0]
    if n_triangles == 0:
        raise ValueError("Unable to read number of triangles as 0")

    triangles = np.empty((n_triangles, 3, 3), dtype=np.float32)
    normals = np.empty((n_triangles, 3), dtype=np.float32)

    for idx in range(n_triangles):
        content = buff.read(50)
        normal = np.frombuffer(content[0:12], dtype=np.float32)
        triangle = np.frombuffer(content[12:48], dtype=np.float32).reshape((3, 3))
        triangles[idx] = triangle
        normals[idx] = normal

    return triangles, normals


def export_stl(filename: pathlib.Path, triangle_vertices: np.ndarray, normals: np.ndarray):
    """Export geometry in STL format

    Args:
        filename (pathlib.Path): Filename to save to.
        triangle_vertices (np.ndarray): Array of the vertices of the triangles.
        normals (np.ndarray): Array of triangles normals.
    """
    # 80 bytes header
    HEADER = b"\x00" * 80

    n_triangles = len(triangle_vertices)
    # STL content is header + uint32 + 50 bytes per triangle
    stl_content = bytearray(len(HEADER) + 4 + 50 * n_triangles)

    # Add header and number of triangles
    stl_content[: len(HEADER)] = HEADER
    stl_content[len(HEADER) : len(HEADER) + 4] = np.uint32(n_triangles).tobytes()

    idx_curr = len(HEADER) + 4

    for idx, t_verts in enumerate(triangle_vertices):
        p0, p1, p2 = [t_verts[0], t_verts[1], t_verts[2]]
        normal = normals[idx]
        t_arr = np.array([normal, p0, p1, p2], dtype=np.float32)

        # Add triangles to STL content and 2 bytes padding
        stl_content[idx_curr : idx_curr + 50] = t_arr.tobytes("C") + b"\x00\x00"
        idx_curr += 50

    create_folders_for_file(filename)

    with open(filename, "wb") as f:
        f.write(bytes(stl_content))
