"""Tests for ByCylindricalGrouping."""

from __future__ import annotations

import numpy as np
import pytest
from lnas import LnasFormat, LnasGeometry

from cfdmod.geometry import (
    ByCylindricalGrouping,
    BySurfaceGrouping,
    apply_groupings,
)


def _tri_at(centroid: tuple[float, float, float], delta: float = 0.05) -> np.ndarray:
    """Return a (1, 3, 3) triangle whose centroid equals ``centroid``."""
    cx, cy, cz = centroid
    v = np.array(
        [
            [cx + delta, cy - delta, cz],
            [cx - delta, cy + delta, cz + delta],
            [cx, cy, cz - delta],
        ],
        dtype=np.float32,
    )
    return v[None, :, :]


@pytest.fixture
def ring_mesh() -> LnasFormat:
    """Four triangles arranged at angles 0, 90, 180, 270 deg around z at r=1."""
    centroids = [(1.0, 0.0, 0.0), (0.0, 1.0, 0.0), (-1.0, 0.0, 0.0), (0.0, -1.0, 0.0)]
    triangles = np.concatenate([_tri_at(c) for c in centroids], axis=0)  # (4, 3, 3)

    n = triangles.shape[0]
    vertices = triangles.reshape((n * 3, 3)).astype(np.float32)
    tri_idx = np.arange(n * 3, dtype=np.uint32).reshape((n, 3))
    geometry = LnasGeometry(vertices=vertices, triangles=tri_idx)
    surfaces = {"ring": np.arange(n, dtype=np.uint32)}
    return LnasFormat(version="v1.0", geometry=geometry, surfaces=surfaces)


def test_four_quadrant_sectors(ring_mesh):
    spec = ByCylindricalGrouping(
        origin=(0.0, 0.0, 0.0),
        axis="z",
        theta_intervals_deg=[0.0, 90.0, 180.0, 270.0, 360.0],
        name_template="t{it}",
    )
    res = apply_groupings(ring_mesh, [spec])
    # Centroid (1,0,0): theta=0   -> t0
    # Centroid (0,1,0): theta=90  -> t1
    # Centroid (-1,0,0): theta=180-> t2
    # Centroid (0,-1,0): theta=270-> t3
    assert set(res.groups) == {"t0", "t1", "t2", "t3"}
    assert res.groups["t0"].tolist() == [0]
    assert res.groups["t1"].tolist() == [1]
    assert res.groups["t2"].tolist() == [2]
    assert res.groups["t3"].tolist() == [3]


def test_radial_split(ring_mesh):
    # Two radial bands: [0, 0.5) and [0.5, 2.0). Ring centroids at r=1 fall
    # in the outer band; nothing in the inner.
    spec = ByCylindricalGrouping(
        origin=(0.0, 0.0, 0.0),
        axis="z",
        r_intervals=[0.0, 0.5, 2.0],
        name_template="r{ir}",
    )
    res = apply_groupings(ring_mesh, [spec])
    assert set(res.groups) == {"r1"}
    assert sorted(res.groups["r1"].tolist()) == [0, 1, 2, 3]


def test_axis_x_uses_yz_plane_for_theta():
    # Axis x: theta is measured from +y toward +z. Place a centroid at
    # (5, 1, 0) -> theta=0 in the yz plane; (5, 0, 1) -> theta=90.
    triangles = np.concatenate([_tri_at((5.0, 1.0, 0.0)), _tri_at((5.0, 0.0, 1.0))], axis=0)
    n = triangles.shape[0]
    vertices = triangles.reshape((n * 3, 3)).astype(np.float32)
    tri_idx = np.arange(n * 3, dtype=np.uint32).reshape((n, 3))
    geometry = LnasGeometry(vertices=vertices, triangles=tri_idx)
    mesh = LnasFormat(
        version="v1.0",
        geometry=geometry,
        surfaces={"S": np.arange(n, dtype=np.uint32)},
    )
    spec = ByCylindricalGrouping(
        origin=(5.0, 0.0, 0.0),
        axis="x",
        theta_intervals_deg=[0.0, 90.0, 180.0],
        name_template="t{it}",
    )
    res = apply_groupings(mesh, [spec])
    assert res.groups["t0"].tolist() == [0]
    assert res.groups["t1"].tolist() == [1]


def test_axial_split_with_z_axis(ring_mesh):
    # Move two of the ring's centroids upward via a fresh mesh: just slice
    # the original ring (axial = z = 0) into two bands; everything in the
    # lower one.
    spec = ByCylindricalGrouping(
        origin=(0.0, 0.0, 0.0),
        axis="z",
        axial_intervals=[-1.0, 0.5, 2.0],
        name_template="z{iz}",
    )
    res = apply_groupings(ring_mesh, [spec])
    assert set(res.groups) == {"z0"}
    assert sorted(res.groups["z0"].tolist()) == [0, 1, 2, 3]


def test_combined_r_theta_axial_product(ring_mesh):
    spec = ByCylindricalGrouping(
        origin=(0.0, 0.0, 0.0),
        axis="z",
        r_intervals=[0.0, 2.0],
        theta_intervals_deg=[0.0, 180.0, 360.0],
        axial_intervals=[-1.0, 1.0],
        name_template="c{ir}_{it}_{iz}",
    )
    res = apply_groupings(ring_mesh, [spec])
    # 1 r-cell, 2 theta-cells, 1 axial-cell -> 2 named cells.
    # theta in [0,180): centroids at 0 deg (tri 0) and 90 deg (tri 1).
    # theta in [180,360): centroids at 180 deg (tri 2) and 270 deg (tri 3).
    assert sorted(res.groups["c0_0_0"].tolist()) == [0, 1]
    assert sorted(res.groups["c0_1_0"].tolist()) == [2, 3]


def test_restrict_to_filters_candidates(ring_mesh):
    surfs = BySurfaceGrouping(sets={"half": ["ring"]})
    spec = ByCylindricalGrouping(
        origin=(0.0, 0.0, 0.0),
        axis="z",
        theta_intervals_deg=[0.0, 180.0],
        restrict_to=["half"],
        name_template="h{it}",
    )
    res = apply_groupings(ring_mesh, [surfs, spec])
    # Only triangles in [0, 180) deg pass: tris 0 (0 deg) and 1 (90 deg).
    assert sorted(res.groups["h0"].tolist()) == [0, 1]


def test_validation_rejects_negative_radius():
    with pytest.raises(ValueError):
        ByCylindricalGrouping(origin=(0, 0, 0), r_intervals=[-1.0, 1.0])


def test_validation_rejects_theta_out_of_range():
    with pytest.raises(ValueError):
        ByCylindricalGrouping(origin=(0, 0, 0), theta_intervals_deg=[-10.0, 10.0])
    with pytest.raises(ValueError):
        ByCylindricalGrouping(origin=(0, 0, 0), theta_intervals_deg=[0.0, 400.0])
