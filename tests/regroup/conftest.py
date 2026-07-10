"""Shared fixtures and helpers for ``tests/regroup``."""

from __future__ import annotations

import pathlib

import h5py
import numpy as np
import pytest
from lnas import LnasFormat, LnasGeometry
from lnas import fmt as _lnas_fmt

REPO_ROOT = pathlib.Path(__file__).resolve().parents[2]
PRESSURE_DATA = REPO_ROOT / "fixtures" / "tests" / "pressure" / "data"
GALPAO_CP_H5 = PRESSURE_DATA / "cp_t.normalized.h5"


def _grid_mesh(
    nx: int,
    ny: int,
    *,
    cell_size: float = 1.0,
    z: float = 0.0,
    x_offset: float = 0.0,
    y_offset: float = 0.0,
) -> LnasGeometry:
    """A simple ``nx*ny``-cell grid of two triangles per cell, in the XY plane."""
    n_verts_x = nx + 1
    n_verts_y = ny + 1
    xs = np.arange(n_verts_x, dtype=np.float32) * cell_size + x_offset
    ys = np.arange(n_verts_y, dtype=np.float32) * cell_size + y_offset
    xv, yv = np.meshgrid(xs, ys, indexing="xy")
    verts = np.stack(
        [xv.flatten(), yv.flatten(), np.full(xv.size, z, dtype=np.float32)], axis=1
    ).astype(np.float32)
    tris = []
    for j in range(ny):
        for i in range(nx):
            v0 = j * n_verts_x + i
            v1 = j * n_verts_x + (i + 1)
            v2 = (j + 1) * n_verts_x + i
            v3 = (j + 1) * n_verts_x + (i + 1)
            tris.append([v0, v1, v2])
            tris.append([v1, v3, v2])
    triangles = np.array(tris, dtype=np.uint32)
    return LnasGeometry(vertices=verts, triangles=triangles)


def _join_meshes(meshes: list[LnasGeometry]) -> LnasGeometry:
    all_verts = []
    all_tris = []
    offset = 0
    for g in meshes:
        all_verts.append(g.vertices)
        all_tris.append(g.triangles + offset)
        offset += g.vertices.shape[0]
    verts = np.concatenate(all_verts, axis=0).astype(np.float32)
    tris = np.concatenate(all_tris, axis=0).astype(np.uint32)
    return LnasGeometry(vertices=verts, triangles=tris)


@pytest.fixture
def small_mesh() -> LnasFormat:
    """4-triangle planar mesh in the XY plane (centroids span [0,2] x [0,2])."""
    geom = _grid_mesh(2, 2, cell_size=1.0)
    return LnasFormat(
        version=_lnas_fmt._CURRENT_VERSION,
        geometry=geom,
        surfaces={"all": np.arange(geom.triangles.shape[0], dtype=np.uint32)},
    )


@pytest.fixture
def two_container_mesh() -> LnasFormat:
    """Two disjoint planar grids (containers) at different positions and sizes.

    - Container A: 4m x 6m grid (8 cells x 12 cells -> 192 triangles), at origin.
    - Container B: 2m x 3m grid (4 cells x 6 cells -> 48 triangles), offset.

    Total 240 triangles in two connected components, axis-aligned.
    """
    a = _grid_mesh(4, 6, cell_size=1.0, x_offset=0.0, y_offset=0.0, z=0.0)
    b = _grid_mesh(2, 3, cell_size=1.0, x_offset=20.0, y_offset=20.0, z=0.0)
    joined = _join_meshes([a, b])
    return LnasFormat(
        version=_lnas_fmt._CURRENT_VERSION,
        geometry=joined,
        surfaces={"all": np.arange(joined.triangles.shape[0], dtype=np.uint32)},
    )


def _bumpy_grid(
    nx: int,
    ny: int,
    *,
    cell_size: float = 1.0,
    amplitude: float = 0.6,
) -> LnasGeometry:
    """A grid whose height varies as a product of sines.

    Unlike the planar grids, every triangle has a normal with non-trivial
    x/y/z components and spans a finite range on all three axes, so cuts
    along x, y and z all exercise the straddling (slicing) path.
    """
    n_verts_x = nx + 1
    n_verts_y = ny + 1
    xs = np.arange(n_verts_x, dtype=np.float64) * cell_size
    ys = np.arange(n_verts_y, dtype=np.float64) * cell_size
    xv, yv = np.meshgrid(xs, ys, indexing="xy")
    kx = 2.0 * np.pi / (nx * cell_size)
    ky = 2.0 * np.pi / (ny * cell_size)
    zv = amplitude * np.sin(kx * xv) * np.sin(ky * yv)
    verts = np.stack([xv.flatten(), yv.flatten(), zv.flatten()], axis=1).astype(np.float32)
    tris = []
    for j in range(ny):
        for i in range(nx):
            v0 = j * n_verts_x + i
            v1 = j * n_verts_x + (i + 1)
            v2 = (j + 1) * n_verts_x + i
            v3 = (j + 1) * n_verts_x + (i + 1)
            tris.append([v0, v1, v2])
            tris.append([v1, v3, v2])
    triangles = np.array(tris, dtype=np.uint32)
    return LnasGeometry(vertices=verts, triangles=triangles)


@pytest.fixture
def curved_mesh() -> LnasFormat:
    """A non-planar bumpy grid whose triangles straddle x, y and z planes."""
    geom = _bumpy_grid(12, 12, cell_size=1.0, amplitude=0.6)
    return LnasFormat(
        version=_lnas_fmt._CURRENT_VERSION,
        geometry=geom,
        surfaces={"all": np.arange(geom.triangles.shape[0], dtype=np.uint32)},
    )


def make_synthetic_cp_h5(
    h5_path: pathlib.Path,
    n_triangles: int,
    n_steps: int,
    *,
    group: str = "cp",
    seed: int = 0,
) -> tuple[np.ndarray, np.ndarray]:
    """Write a synthetic per-triangle timeseries; return (data, time_steps).

    Geometry datasets are written as small placeholders (the regroup
    pipeline reads its own geometry from the LnasFormat input).
    """
    rng = np.random.default_rng(seed)
    data = rng.normal(size=(n_steps, n_triangles)).astype(np.float64)
    times = np.arange(n_steps, dtype=np.float64)
    h5_path.parent.mkdir(parents=True, exist_ok=True)
    with h5py.File(h5_path, "w") as f:
        f.create_dataset("Triangles", data=np.zeros((n_triangles, 3), dtype=np.int32))
        f.create_dataset("Geometry", data=np.zeros((1, 3), dtype=np.float64))
        grp = f.create_group(group)
        for t, row in zip(times, data):
            grp.create_dataset(f"t{t}", data=row)
        meta = f.create_group("meta")
        meta.create_dataset("time_steps", data=times)
        meta.create_dataset("time_normalized", data=times)
    return data, times
