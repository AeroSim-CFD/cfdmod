"""Shared fixtures for grouping tests.

Builds tiny synthetic LnasFormat meshes so tests don't depend on the
heavier pressure-test fixtures.
"""

from __future__ import annotations

import numpy as np
import pytest
from lnas import LnasFormat, LnasGeometry


def _square(z: float, x_offset: float = 0.0, y_offset: float = 0.0, size: float = 1.0) -> np.ndarray:
    """Two triangles tiling [0,size]x[0,size] at height z, offset in x/y."""
    s = size
    v = np.array(
        [
            [x_offset + 0, y_offset + 0, z],
            [x_offset + s, y_offset + 0, z],
            [x_offset + s, y_offset + s, z],
            [x_offset + 0, y_offset + s, z],
        ],
        dtype=np.float32,
    )
    return np.stack([v[[0, 1, 2]], v[[0, 2, 3]]], axis=0)


@pytest.fixture
def two_square_mesh() -> LnasFormat:
    """Two coplanar squares (4 triangles total) in two named surfaces.

    Surface ``A`` covers x in [0,1], surface ``B`` covers x in [2,3].
    Both at z=0. The two surfaces are spatially disjoint along x.
    """
    sq_a = _square(z=0.0, x_offset=0.0, y_offset=0.0)
    sq_b = _square(z=0.0, x_offset=2.0, y_offset=0.0)
    triangles = np.concatenate([sq_a, sq_b], axis=0)  # (4, 3, 3)

    n = triangles.shape[0]
    vertices = triangles.reshape((n * 3, 3)).astype(np.float32)
    tri_idx = np.arange(n * 3, dtype=np.uint32).reshape((n, 3))
    geometry = LnasGeometry(vertices=vertices, triangles=tri_idx)

    surfaces = {
        "A": np.array([0, 1], dtype=np.uint32),
        "B": np.array([2, 3], dtype=np.uint32),
    }
    return LnasFormat(version="v1.0", geometry=geometry, surfaces=surfaces)


@pytest.fixture
def grid_mesh() -> LnasFormat:
    """3x1 grid of unit squares in x, single surface ``S``.

    Centroids land at x = 0.5, 1.5, 2.5; y = 0.5; z = 0. Useful for
    exercising x_intervals binning.
    """
    sqs = [_square(z=0.0, x_offset=float(i)) for i in range(3)]
    triangles = np.concatenate(sqs, axis=0)  # (6, 3, 3)

    n = triangles.shape[0]
    vertices = triangles.reshape((n * 3, 3)).astype(np.float32)
    tri_idx = np.arange(n * 3, dtype=np.uint32).reshape((n, 3))
    geometry = LnasGeometry(vertices=vertices, triangles=tri_idx)

    surfaces = {"S": np.arange(n, dtype=np.uint32)}
    return LnasFormat(version="v1.0", geometry=geometry, surfaces=surfaces)


def _face_triangles(face: str) -> np.ndarray:
    """Two triangles tiling a unit-square face of the [0,1]^3 cube.

    Triangles use a vertex order that yields a normal pointing outward
    along the named face direction (one of ``+x``, ``-x``, ``+y``, ...).
    Returns a (2, 3, 3) float32 array.
    """
    # Local frame on each face; orient so cross(v1-v0, v2-v0) points outward.
    layouts: dict[str, list[tuple[float, float, float]]] = {
        "+x": [(1, 0, 0), (1, 1, 0), (1, 1, 1), (1, 0, 1)],
        "-x": [(0, 0, 0), (0, 0, 1), (0, 1, 1), (0, 1, 0)],
        "+y": [(0, 1, 0), (0, 1, 1), (1, 1, 1), (1, 1, 0)],
        "-y": [(0, 0, 0), (1, 0, 0), (1, 0, 1), (0, 0, 1)],
        "+z": [(0, 0, 1), (1, 0, 1), (1, 1, 1), (0, 1, 1)],
        "-z": [(0, 0, 0), (0, 1, 0), (1, 1, 0), (1, 0, 0)],
    }
    quad = np.array(layouts[face], dtype=np.float32)
    return np.stack([quad[[0, 1, 2]], quad[[0, 2, 3]]], axis=0)


@pytest.fixture
def cube_mesh() -> LnasFormat:
    """Unit cube [0,1]^3 with two triangles per face (12 total).

    Triangle order is +x, -x, +y, -y, +z, -z (two triangles per face,
    consecutive). Outward normals match the face direction.
    """
    faces = ["+x", "-x", "+y", "-y", "+z", "-z"]
    triangles = np.concatenate([_face_triangles(f) for f in faces], axis=0)  # (12, 3, 3)

    n = triangles.shape[0]
    vertices = triangles.reshape((n * 3, 3)).astype(np.float32)
    tri_idx = np.arange(n * 3, dtype=np.uint32).reshape((n, 3))
    geometry = LnasGeometry(vertices=vertices, triangles=tri_idx)
    surfaces = {"cube": np.arange(n, dtype=np.uint32)}
    return LnasFormat(version="v1.0", geometry=geometry, surfaces=surfaces)
