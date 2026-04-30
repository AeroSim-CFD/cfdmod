from __future__ import annotations

import pathlib

import lnas
import numpy as np

from cfdmod.io.geometry.STL import export_stl, read_stl
from cfdmod.loft.functions import generate_loft_surface


def test_ground():
    input_path = pathlib.Path("fixtures/tests/loft/complex_terrain.stl")
    output_path = pathlib.Path("output/complex_loft.stl")

    triangles, normals = read_stl(input_path)
    geom = lnas.LnasFormat.from_triangles(triangles=triangles, normals=normals).geometry

    loft_geom = generate_loft_surface(
        geom=geom,
        loft_radius=1500.0,
        loft_z_pos=100.0,
    )

    loft_tris = loft_geom.triangle_vertices
    u = loft_tris[:, 1] - loft_tris[:, 0]
    v = loft_tris[:, 2] - loft_tris[:, 0]
    loft_normals = np.cross(u, v)

    export_stl(output_path, loft_tris, loft_normals)
