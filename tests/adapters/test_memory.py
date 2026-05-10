"""Unit tests for :class:`MemoryFieldStore` and :class:`MemoryStorage`."""

from __future__ import annotations

import numpy as np
import pytest

from cfdmod.adapters.memory import MemoryFieldStore, MemoryStorage
from cfdmod.core import (
    ElementMeta,
    SurfaceDataSource,
    TimeAxis,
    Topology,
)


def test_with_field_does_not_mutate_predecessor():
    store = MemoryFieldStore({"a": np.array([1.0, 2.0, 3.0])})
    store2 = store.with_field("b", np.array([10.0, 20.0, 30.0]))
    assert "b" not in list(store.keys())
    assert "b" in list(store2.keys())


def test_partial_write_into_existing_field():
    store = MemoryFieldStore({"p": np.zeros((4, 5))})
    store.write("p", np.full((2, 5), 7.0), element_slice=slice(1, 3))
    arr = store.read("p")
    assert np.allclose(arr[1:3], 7.0)
    assert np.allclose(arr[0], 0.0)
    assert np.allclose(arr[3], 0.0)


def test_partial_write_into_missing_field_raises():
    store = MemoryFieldStore()
    with pytest.raises(KeyError):
        store.write("p", np.zeros(3), element_slice=slice(0, 1))


def test_read_with_fancy_indexing():
    store = MemoryFieldStore({"p": np.arange(20).reshape(5, 4)})
    elements = np.array([0, 2, 4])
    out = store.read("p", elements=elements)
    assert out.shape == (3, 4)
    assert np.array_equal(out[0], [0, 1, 2, 3])
    assert np.array_equal(out[1], [8, 9, 10, 11])


def test_read_rejects_both_element_slice_and_elements():
    store = MemoryFieldStore({"p": np.zeros((3, 4))})
    with pytest.raises(ValueError):
        store.read("p", element_slice=slice(0, 2), elements=np.array([0, 1]))


def test_memory_storage_round_trip_data_source():
    verts = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0]], dtype=np.float64)
    tris = np.array([[0, 1, 2]], dtype=np.int32)
    ds = SurfaceDataSource(
        time=TimeAxis(initial_time=0.0, timestep_size=0.1, n_timesteps=2),
        topology=Topology.triangles(tris, verts),
        elements=ElementMeta(),
        fields=MemoryFieldStore({"p": np.array([[1.0, 2.0]])}),
    )
    storage = MemoryStorage()
    storage.write_data_source("foo", ds)
    assert "foo" in storage
    assert storage.read_data_source("foo") is ds
    with pytest.raises(KeyError):
        storage.read_data_source("missing")
