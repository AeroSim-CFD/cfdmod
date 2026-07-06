"""Property-based storage round-trip tests for the v3 data-source layer.

The highest-value invariant of the storage seam is a literal contract:
``write -> read`` returns an equal source. These tests assert it generatively
over many randomly-generated sources instead of the single hand-written shapes
in ``tests/adapters/test_conformance.py``.

Scope of "equal" differs by backend:

- :class:`MemoryStorage` keeps the whole frozen object, so everything survives
  (topology, time axis, element metadata, fields).
- :class:`XdmfH5Storage` persists only the v2 byte layout -- triangles/geometry,
  the affine time axis, and the field arrays. Element metadata, groupings, and
  ``attrs`` are deliberately *not* on disk, so the round-trip equality here is
  over the persisted subset. ``n_timesteps == 1`` cannot carry a timestep size
  (a single sample has no delta), so that assertion is skipped for that case.

These tests are marked ``property`` so a nightly job can rerun them with a
larger ``max_examples`` budget.
"""

from __future__ import annotations

import pathlib

import numpy as np
import pytest
from hypothesis import HealthCheck, given, settings

from cfdmod.adapters.memory import MemoryStorage
from cfdmod.adapters.xdmf_h5 import XdmfH5Storage
from cfdmod.core.data_source import DataSource, PointsDataSource
from tests import strategies as sty

pytestmark = pytest.mark.property


def _key_for(ds: DataSource) -> str:
    """A storage key whose prefix makes the H5 adapter read back the right kind."""
    return "points.gen" if isinstance(ds, PointsDataSource) else "cp_t.gen"


def _assert_topology_equal(a: DataSource, b: DataSource) -> None:
    # GroupsDataSource carries no topology of its own (it is chained to a
    # parent); both sides must agree on that.
    assert (a.topology is None) == (b.topology is None)
    if a.topology is None:
        return
    assert a.topology.cell_type == b.topology.cell_type
    assert np.array_equal(a.topology.connectivity, b.topology.connectivity)
    assert np.allclose(a.topology.vertices, b.topology.vertices)


def _assert_fields_equal(a: DataSource, b: DataSource) -> None:
    assert sorted(a.fields.keys()) == sorted(b.fields.keys())
    for name in a.fields.keys():
        arr_a = a.fields.read(name)
        arr_b = b.fields.read(name)
        assert arr_a.shape == arr_b.shape, f"shape mismatch for field {name!r}"
        assert np.allclose(arr_a, arr_b), f"values differ for field {name!r}"


def _assert_persisted_equal(a: DataSource, b: DataSource) -> None:
    """Equality over what any backend must preserve: kind, topology, time, fields."""
    assert a.kind == b.kind
    _assert_topology_equal(a, b)
    assert a.time.n_timesteps == b.time.n_timesteps
    # A time-aggregated source has no time axis: the H5 layout stores no
    # meta/time_steps, so the affine origin (initial_time) is deliberately not
    # preserved -- only "is aggregated" is meaningful. Compare the origin/step
    # only when there actually is a time axis.
    if not a.time.is_time_aggregated:
        assert np.isclose(a.time.initial_time, b.time.initial_time)
        # A single-sample axis has no delta to store; the H5 adapter reads it
        # back as 1.0 by construction, so only compare the step when n >= 2.
        if a.time.n_timesteps >= 2:
            assert np.isclose(a.time.timestep_size, b.time.timestep_size)
    _assert_fields_equal(a, b)


# ---------------------------------------------------------------------------
# MemoryStorage: the whole object survives.
# ---------------------------------------------------------------------------


@given(ds=sty.data_sources())
def test_memory_roundtrip_preserves_source(ds: DataSource) -> None:
    storage = MemoryStorage()
    storage.write_data_source("k", ds)
    back = storage.read_data_source("k")
    assert type(back) is type(ds)
    _assert_persisted_equal(ds, back)
    # Memory keeps the element metadata too (it is not serialised away).
    assert np.allclose(back.elements.position, ds.elements.position)
    assert np.allclose(back.elements.area, ds.elements.area)


# ---------------------------------------------------------------------------
# XdmfH5Storage: the persisted subset survives a real file round-trip.
# ---------------------------------------------------------------------------


@settings(suppress_health_check=[HealthCheck.function_scoped_fixture])
@given(ds=sty.roundtrippable_data_sources())
def test_xdmf_h5_roundtrip_preserves_persisted_subset(
    ds: DataSource, tmp_path: pathlib.Path
) -> None:
    storage = XdmfH5Storage(tmp_path)
    key = _key_for(ds)
    storage.write_data_source(key, ds)
    back = storage.read_data_source(key)
    _assert_persisted_equal(ds, back)


# ---------------------------------------------------------------------------
# Backend equivalence: the two backends agree on the same source.
# ---------------------------------------------------------------------------


@settings(suppress_health_check=[HealthCheck.function_scoped_fixture])
@given(ds=sty.roundtrippable_data_sources())
def test_backends_agree(ds: DataSource, tmp_path: pathlib.Path) -> None:
    key = _key_for(ds)
    mem = MemoryStorage()
    h5 = XdmfH5Storage(tmp_path)
    mem.write_data_source(key, ds)
    h5.write_data_source(key, ds)
    from_mem = mem.read_data_source(key)
    from_h5 = h5.read_data_source(key)
    _assert_persisted_equal(from_mem, from_h5)
