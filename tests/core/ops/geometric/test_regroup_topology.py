"""Unit tests for the regroup_topology op."""

from __future__ import annotations

import pathlib

import numpy as np
import pytest
from lnas import LnasFormat, LnasGeometry

from cfdmod.adapters.memory import MemoryFieldStore
from cfdmod.core import (
    ElementMeta,
    GroupsDataSource,
    SurfaceDataSource,
    TimeAxis,
    Topology,
)
from cfdmod.core.ops.geometric import (
    RegroupTopologyParams,
    regroup_topology,
)
from cfdmod.geometry.grouping import (
    BySizeRoundedPerComponent,
    BySurfaceGrouping,
    ByZoningGrouping,
)


def _two_body_lnas(tmp_path: pathlib.Path) -> tuple[pathlib.Path, LnasFormat]:
    """Two square surfaces (4 triangles), spatially disjoint along x."""
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


def _surface_ds(lnas: LnasFormat, pressure: np.ndarray) -> SurfaceDataSource:
    return SurfaceDataSource(
        time=TimeAxis(initial_time=0.0, timestep_size=1.0, n_timesteps=pressure.shape[1]),
        topology=Topology.triangles(
            lnas.geometry.triangles.astype(np.int32),
            lnas.geometry.vertices.astype(np.float64),
        ),
        elements=ElementMeta(),
        fields=MemoryFieldStore({"pressure": pressure}),
    )


def test_area_weighted_mean_per_surface_group(tmp_path):
    """Two surfaces -> two groups; area-weighted mean over equal-area triangles."""
    mesh_path, lnas = _two_body_lnas(tmp_path)
    pressure = np.array([[1.0, 2.0], [3.0, 4.0], [10.0, 20.0], [30.0, 40.0]])
    ds = _surface_ds(lnas, pressure)

    out = regroup_topology(
        ds,
        RegroupTopologyParams(
            mesh=str(mesh_path),
            groupings=[BySurfaceGrouping(sets={"left": ["A"], "right": ["B"]})],
            aggregation="area_weighted_mean",
        ),
    )

    assert isinstance(out, GroupsDataSource)
    assert out.n_elements == 2
    np.testing.assert_allclose(out.fields.read("pressure")[0], [2.0, 3.0])
    np.testing.assert_allclose(out.fields.read("pressure")[1], [20.0, 30.0])
    assert out.groupings["regroup"].label(0) == "left"
    assert out.groupings["regroup"].label(1) == "right"


def test_sum_aggregation(tmp_path):
    mesh_path, lnas = _two_body_lnas(tmp_path)
    pressure = np.array([[1.0], [3.0], [10.0], [30.0]])
    ds = _surface_ds(lnas, pressure)

    out = regroup_topology(
        ds,
        RegroupTopologyParams(
            mesh=str(mesh_path),
            groupings=[BySurfaceGrouping(sets={"left": ["A"], "right": ["B"]})],
            aggregation="sum",
        ),
    )
    np.testing.assert_allclose(out.fields.read("pressure")[0], [4.0])
    np.testing.assert_allclose(out.fields.read("pressure")[1], [40.0])


def test_overlapping_groups_raise(tmp_path):
    """A triangle reaching two output groups must fail; semantics require single membership."""
    mesh_path, lnas = _two_body_lnas(tmp_path)
    pressure = np.zeros((4, 1))
    ds = _surface_ds(lnas, pressure)

    with pytest.raises(ValueError, match="multiple output groups"):
        regroup_topology(
            ds,
            RegroupTopologyParams(
                mesh=str(mesh_path),
                groupings=[BySurfaceGrouping(sets={"all1": ["A", "B"], "all2": ["A", "B"]})],
                aggregation="sum",
            ),
        )


def test_chain_with_size_rounded_expands_per_component(tmp_path):
    """BySizeRoundedPerComponent splits each parent surface independently."""
    mesh_path, lnas = _two_body_lnas(tmp_path)
    pressure = np.arange(4 * 3, dtype=np.float64).reshape(4, 3)
    ds = _surface_ds(lnas, pressure)

    out = regroup_topology(
        ds,
        RegroupTopologyParams(
            mesh=str(mesh_path),
            groupings=[
                BySurfaceGrouping(sets={"left": ["A"], "right": ["B"]}),
                BySizeRoundedPerComponent(target_size_x=10.0, name_template="{parent}_c{idx}"),
            ],
            aggregation="area_weighted_mean",
        ),
    )

    labels = {out.groupings["regroup"].label(i) for i in range(out.n_elements)}
    assert labels == {"left_c0", "right_c0"}


def test_explicit_zoning_grouping(tmp_path):
    """Single ByZoningGrouping divides parent into two cells along x."""
    mesh_path, lnas = _two_body_lnas(tmp_path)
    pressure = np.array([[1.0], [2.0], [3.0], [4.0]])
    ds = _surface_ds(lnas, pressure)

    out = regroup_topology(
        ds,
        RegroupTopologyParams(
            mesh=str(mesh_path),
            groupings=[
                ByZoningGrouping(
                    x_intervals=[0.0, 1.5, 3.5],
                    name_template="cell_{ix}",
                )
            ],
            aggregation="area_weighted_mean",
        ),
    )
    assert out.n_elements == 2


def test_empty_groupings_raises(tmp_path):
    mesh_path, lnas = _two_body_lnas(tmp_path)
    ds = _surface_ds(lnas, np.zeros((4, 1)))
    with pytest.raises(ValueError, match="empty"):
        regroup_topology(ds, RegroupTopologyParams(mesh=str(mesh_path), groupings=[]))
