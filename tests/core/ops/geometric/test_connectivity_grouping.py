"""Tests for the connectivity_grouping op (topological Cf/Cm region split).

Covers the op adapter directly and an integration/regression check that a
connectivity grouping feeds the Cm recipe and, on axis-aligned geometry,
produces the same partition as the rectangular zoning_grouping.
"""

from __future__ import annotations

import pathlib

import numpy as np
import pytest
from lnas import LnasFormat, LnasGeometry

from cfdmod.adapters.memory import MemoryFieldStore
from cfdmod.core import (
    ElementMeta,
    SurfaceDataSource,
    TimeAxis,
    Topology,
)
from cfdmod.core.ops.geometric import (
    ConnectivityGroupingParams,
    ZoningGroupingParams,
    connectivity_grouping,
    zoning_grouping,
)
from cfdmod.core.recipes import CmRecipeConfig, cm_pipeline


def _two_body_lnas(tmp_path: pathlib.Path) -> tuple[pathlib.Path, LnasFormat]:
    """Two unit squares (2 triangles each) spatially disjoint along x.

    Each square is one connected component (its two triangles share an
    edge); the squares share no vertices, so connectivity yields exactly
    two components. The layout is axis-aligned, so a rectangular x-split
    yields the same two regions -- the basis of the regression check.
    """
    verts = np.array(
        [
            [0, 0, 0],
            [1, 0, 0],
            [1, 1, 0],
            [0, 1, 0],
            [2, 0, 0],
            [3, 0, 0],
            [3, 1, 0],
            [2, 1, 0],
        ],
        dtype=np.float32,
    )
    tris = np.array(
        [
            [0, 1, 2],
            [0, 2, 3],
            [4, 5, 6],
            [4, 6, 7],
        ],
        dtype=np.uint32,
    )
    geom = LnasGeometry(vertices=verts, triangles=tris)
    surfaces = {
        "A": np.array([0, 1], dtype=np.uint32),
        "B": np.array([2, 3], dtype=np.uint32),
    }
    lnas = LnasFormat(version="v0.5.0", geometry=geom, surfaces=surfaces)
    path = tmp_path / "two_body.lnas"
    lnas.to_file(path)
    return path, lnas


def _surface_ds(lnas: LnasFormat, n_timesteps: int = 1) -> SurfaceDataSource:
    n_tri = lnas.geometry.triangles.shape[0]
    return SurfaceDataSource(
        time=TimeAxis(initial_time=0.0, timestep_size=1.0, n_timesteps=n_timesteps),
        topology=Topology.triangles(
            lnas.geometry.triangles.astype(np.int32),
            lnas.geometry.vertices.astype(np.float64),
        ),
        elements=ElementMeta(area=np.ones(n_tri)),
        fields=MemoryFieldStore({"cp": np.zeros((n_tri, n_timesteps))}),
    )


def test_two_disjoint_bodies_split_into_two_regions(tmp_path):
    mesh_path, lnas = _two_body_lnas(tmp_path)
    ds = _surface_ds(lnas)

    out = connectivity_grouping(ds, ConnectivityGroupingParams(mesh=str(mesh_path)))

    grouping = out.groupings["body"]
    # Two equal-size components, ordered by first triangle index -> cc0, cc1.
    np.testing.assert_array_equal(grouping.indices, [0, 0, 1, 1])
    assert grouping.id_to_label == {0: "cc0", 1: "cc1"}
    assert -1 not in grouping.indices


def test_min_triangles_drops_debris_leaving_sentinel(tmp_path):
    """A lone triangle below min_triangles stays ungrouped (-1)."""
    mesh_path, lnas = _two_body_lnas(tmp_path)
    # Append a free-floating single triangle (its own component of size 1).
    verts = np.concatenate(
        [lnas.geometry.vertices, np.array([[10, 10, 0], [11, 10, 0], [10, 11, 0]], np.float32)]
    )
    n_v = lnas.geometry.vertices.shape[0]
    tris = np.concatenate(
        [lnas.geometry.triangles, np.array([[n_v, n_v + 1, n_v + 2]], np.uint32)]
    )
    debris = LnasFormat(
        version="v0.5.0",
        geometry=LnasGeometry(vertices=verts, triangles=tris),
        surfaces={"all": np.arange(5, dtype=np.uint32)},
    )
    debris_path = tmp_path / "debris.lnas"
    debris.to_file(debris_path)
    ds = _surface_ds(debris)

    out = connectivity_grouping(
        ds, ConnectivityGroupingParams(mesh=str(debris_path), min_triangles=2)
    )

    grouping = out.groupings["body"]
    # The two size-2 squares survive; the lone triangle (index 4) is dropped.
    np.testing.assert_array_equal(grouping.indices, [0, 0, 1, 1, -1])


def test_element_count_mismatch_raises(tmp_path):
    mesh_path, _ = _two_body_lnas(tmp_path)  # 4-triangle mesh
    # A self-consistent 2-triangle data source paired with the 4-triangle mesh.
    verts = np.array([[0, 0, 0], [1, 0, 0], [1, 1, 0], [0, 1, 0]], dtype=np.float64)
    tris = np.array([[0, 1, 2], [0, 2, 3]], dtype=np.int32)
    ds = SurfaceDataSource(
        time=TimeAxis(initial_time=0.0, timestep_size=1.0, n_timesteps=1),
        topology=Topology.triangles(tris, verts),
        elements=ElementMeta(),
        fields=MemoryFieldStore({"cp": np.zeros((2, 1))}),
    )
    with pytest.raises(ValueError, match="triangles but data source has"):
        connectivity_grouping(ds, ConnectivityGroupingParams(mesh=str(mesh_path)))


def test_connectivity_matches_rectangular_partition_through_cm(tmp_path):
    """Regression: on axis-aligned geometry the connectivity partition
    matches the rectangular zoning one, and both drive the Cm recipe to the
    same per-region result."""
    mesh_path, lnas = _two_body_lnas(tmp_path)
    cm_x = np.array([[1.0], [2.0], [10.0], [20.0]])
    cm_y = np.array([[3.0], [4.0], [30.0], [40.0]])

    base = _surface_ds(lnas)
    base = base.with_field("cm_x", cm_x).with_field("cm_y", cm_y)

    conn = connectivity_grouping(
        base, ConnectivityGroupingParams(mesh=str(mesh_path), name="region")
    )
    # Rectangular x-split at 1.5 separates the two boxes into two bins.
    rect = zoning_grouping(
        base,
        ZoningGroupingParams(mesh=str(mesh_path), x_intervals=[0.0, 1.5, 3.0], name="region"),
    )

    # Same partition (identical per-triangle region ids) on this geometry.
    np.testing.assert_array_equal(
        conn.groupings["region"].indices, rect.groupings["region"].indices
    )

    cfg = CmRecipeConfig(grouping="region", directions=["x", "y"])
    out_conn = cm_pipeline(cfg)(conn)
    out_rect = cm_pipeline(cfg)(rect)

    assert out_conn.n_elements == out_rect.n_elements == 2
    np.testing.assert_allclose(out_conn.fields.read("cm_x"), out_rect.fields.read("cm_x"))
    np.testing.assert_allclose(out_conn.fields.read("cm_y"), out_rect.fields.read("cm_y"))
    # Sum within each region: body A = 1+2 / 3+4; body B = 10+20 / 30+40.
    np.testing.assert_allclose(out_conn.fields.read("cm_x")[:, 0], [3.0, 30.0])
    np.testing.assert_allclose(out_conn.fields.read("cm_y")[:, 0], [7.0, 70.0])
