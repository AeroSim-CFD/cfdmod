"""Unit tests for the five :class:`DataSource` kinds.

Construction tests assert each subclass enforces its topology and time
axis invariants. Functional-update tests assert that ``with_*`` methods
return a fresh instance and the predecessor is unchanged.
"""

from __future__ import annotations

import numpy as np
import pytest

from cfdmod.core import (
    ElementMeta,
    FieldMeta,
    Grouping,
    GroupsDataSource,
    ModesDataSource,
    PointsDataSource,
    SurfaceDataSource,
    TimeAxis,
    Topology,
    VolumeDataSource,
)
from cfdmod.adapters.memory import MemoryFieldStore


def _surface(n_elements: int = 3, n_timesteps: int = 4) -> SurfaceDataSource:
    verts = np.array(
        [[0, 0, 0], [1, 0, 0], [0, 1, 0], [1, 1, 0], [0.5, 0.5, 0]], dtype=np.float64
    )
    tris = np.array([[0, 1, 2], [1, 3, 2], [2, 3, 4]], dtype=np.int32)[:n_elements]
    pressure = np.arange(n_elements * n_timesteps, dtype=np.float64).reshape(
        n_elements, n_timesteps
    )
    return SurfaceDataSource(
        time=TimeAxis(initial_time=0.0, timestep_size=0.1, n_timesteps=n_timesteps),
        topology=Topology.triangles(tris, verts),
        elements=ElementMeta(),
        fields=MemoryFieldStore({"pressure": pressure}),
        field_meta={"pressure": FieldMeta(name="pressure")},
    )


def test_surface_data_source_round_trips_basic_metadata():
    ds = _surface()
    assert ds.kind == "surface"
    assert ds.n_elements == 3
    assert ds.field_names == ["pressure"]


def test_surface_data_source_rejects_field_with_wrong_n_elements():
    verts = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0]], dtype=np.float64)
    tris = np.array([[0, 1, 2]], dtype=np.int32)
    bad = np.zeros((5, 4))
    with pytest.raises(ValueError):
        SurfaceDataSource(
            time=TimeAxis(initial_time=0.0, timestep_size=0.1, n_timesteps=4),
            topology=Topology.triangles(tris, verts),
            elements=ElementMeta(),
            fields=MemoryFieldStore({"p": bad}),
        )


def test_surface_data_source_rejects_non_triangle_topology():
    with pytest.raises(ValueError):
        SurfaceDataSource(
            time=TimeAxis(initial_time=0.0, timestep_size=0.0, n_timesteps=0),
            topology=Topology.points(np.zeros((2, 3))),
            elements=ElementMeta(),
            fields=MemoryFieldStore(),
        )


def test_points_data_source_requires_point_topology():
    pts = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0]], dtype=np.float64)
    ds = PointsDataSource(
        time=TimeAxis(initial_time=0.0, timestep_size=0.0, n_timesteps=0),
        topology=Topology.points(pts),
        elements=ElementMeta(position=pts),
        fields=MemoryFieldStore(),
    )
    assert ds.n_elements == 3


def test_volume_data_source_requires_cell_topology():
    verts = np.zeros((4, 3))
    cells = np.array([[0, 1, 2, 3]], dtype=np.int32)
    ds = VolumeDataSource(
        time=TimeAxis(initial_time=0.0, timestep_size=0.0, n_timesteps=0),
        topology=Topology(cell_type="cell", connectivity=cells, vertices=verts),
        elements=ElementMeta(),
        fields=MemoryFieldStore(),
    )
    assert ds.kind == "volume"


def test_modes_data_source_must_not_carry_topology():
    ds = ModesDataSource(
        time=TimeAxis(initial_time=0.0, timestep_size=0.1, n_timesteps=10),
        topology=None,
        elements=ElementMeta(),
        fields=MemoryFieldStore({"q": np.zeros((2, 10))}),
    )
    assert ds.kind == "modes"
    with pytest.raises(ValueError):
        ModesDataSource(
            time=TimeAxis(initial_time=0.0, timestep_size=0.0, n_timesteps=0),
            topology=Topology.points(np.zeros((1, 3))),
            elements=ElementMeta(),
            fields=MemoryFieldStore(),
        )


def test_groups_data_source_chains_parent_topology_and_grouping():
    parent = _surface(n_elements=3, n_timesteps=2).topology
    parent_grouping = Grouping(name="region", indices=[0, 0, 1])
    ds = GroupsDataSource(
        time=TimeAxis(initial_time=0.0, timestep_size=1.0, n_timesteps=2),
        topology=None,
        elements=ElementMeta(),
        parent_topology=parent,
        parent_grouping=parent_grouping,
        fields=MemoryFieldStore({"area_mean": np.zeros((2, 2))}),
    )
    assert ds.kind == "groups"
    assert ds.n_elements == 2


def test_groups_data_source_rejects_independent_topology():
    parent = _surface(n_elements=3, n_timesteps=2).topology
    parent_grouping = Grouping(name="region", indices=[0, 0, 1])
    with pytest.raises(ValueError):
        GroupsDataSource(
            time=TimeAxis(initial_time=0.0, timestep_size=0.0, n_timesteps=0),
            topology=Topology.points(np.zeros((2, 3))),
            elements=ElementMeta(),
            parent_topology=parent,
            parent_grouping=parent_grouping,
            fields=MemoryFieldStore(),
        )


def test_with_field_returns_new_instance_and_preserves_old():
    ds = _surface()
    new_field = np.full((3, 4), 7.0)
    ds2 = ds.with_field("ux", new_field)
    assert ds2 is not ds
    assert ds.field_names == ["pressure"]
    assert sorted(ds2.field_names) == ["pressure", "ux"]


def test_field_meta_kept_in_sync_after_with_field():
    ds = _surface()
    ds2 = ds.with_field("ux", np.zeros((3, 4)), meta=FieldMeta(name="ux", unit="m/s"))
    assert ds2.field_meta["ux"].unit == "m/s"


def test_time_aggregated_data_source_requires_1d_fields():
    verts = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0]], dtype=np.float64)
    tris = np.array([[0, 1, 2]], dtype=np.int32)
    with pytest.raises(ValueError):
        SurfaceDataSource(
            time=TimeAxis(initial_time=0.0, timestep_size=0.0, n_timesteps=0),
            topology=Topology.triangles(tris, verts),
            elements=ElementMeta(),
            fields=MemoryFieldStore({"mean": np.zeros((1, 3))}),
        )
