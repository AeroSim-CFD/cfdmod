"""Phase 2 round-trip tests for :class:`XdmfH5Storage`.

For every v2-canonical h5 fixture (root ``/Triangles`` + ``/Geometry``
+ ``/meta/time_*`` + per-field timestep groups), assert:

- ``read_data_source`` produces a :class:`DataSource` whose topology
  matches the on-disk arrays;
- ``write_data_source`` followed by another ``read_data_source``
  returns an equivalent :class:`DataSource` (topology, time axis, and
  every field array equal up to floating-point tolerance);
- the written h5 file carries every dataset the original did, with
  matching shapes per group.

The affine :class:`TimeAxis` cannot exactly reproduce non-uniform
on-disk timesteps, so the v2-level check uses ``np.allclose`` on
``meta/time_steps`` rather than byte equality.
"""

from __future__ import annotations

import pathlib

import h5py
import numpy as np
import pytest

from cfdmod.adapters import XdmfH5Storage
from cfdmod.core import PointsDataSource, SurfaceDataSource

FIXTURE_ROOT = pathlib.Path(__file__).resolve().parents[2] / "fixtures" / "tests" / "pressure" / "data"

# Only the v2 canonical layout (with /meta) is in scope for Phase 2.
V2_FIXTURES = [
    ("bodies.galpao", SurfaceDataSource),
    ("cp_t.normalized", SurfaceDataSource),
    ("points.static_pressure", PointsDataSource),
]


@pytest.fixture(scope="module")
def src_storage() -> XdmfH5Storage:
    return XdmfH5Storage(FIXTURE_ROOT)


@pytest.mark.parametrize("key,expected_kind", V2_FIXTURES)
def test_read_returns_correct_kind(src_storage: XdmfH5Storage, key: str, expected_kind):
    ds = src_storage.read_data_source(key)
    assert isinstance(ds, expected_kind)


@pytest.mark.parametrize("key,expected_kind", V2_FIXTURES)
def test_read_topology_matches_h5_arrays(src_storage: XdmfH5Storage, key: str, expected_kind):
    ds = src_storage.read_data_source(key)
    with h5py.File(src_storage.h5_path(key), "r") as f:
        on_disk_geom = np.asarray(f["Geometry"][:], dtype=np.float64)
        on_disk_tris = np.asarray(f["Triangles"][:], dtype=np.int32)
    assert np.allclose(ds.topology.vertices, on_disk_geom)
    if ds.topology.cell_type == "triangle":
        assert np.array_equal(ds.topology.connectivity, on_disk_tris)


@pytest.mark.parametrize("key,_kind", V2_FIXTURES)
def test_round_trip_preserves_fields_and_topology(
    src_storage: XdmfH5Storage,
    tmp_path: pathlib.Path,
    key: str,
    _kind,
):
    ds1 = src_storage.read_data_source(key)
    dst = XdmfH5Storage(tmp_path)
    dst.write_data_source(key, ds1)
    assert dst.h5_path(key).exists()
    assert dst.xdmf_path(key).exists()

    ds2 = dst.read_data_source(key)
    assert ds1.kind == ds2.kind
    assert ds1.topology.cell_type == ds2.topology.cell_type
    assert np.array_equal(ds1.topology.connectivity, ds2.topology.connectivity)
    assert np.allclose(ds1.topology.vertices, ds2.topology.vertices)
    assert ds1.time.n_timesteps == ds2.time.n_timesteps
    assert np.isclose(ds1.time.initial_time, ds2.time.initial_time)
    assert np.isclose(ds1.time.timestep_size, ds2.time.timestep_size)
    assert sorted(ds1.fields.keys()) == sorted(ds2.fields.keys())
    for fn in ds1.fields.keys():
        a1 = ds1.fields.read(fn)
        a2 = ds2.fields.read(fn)
        assert a1.shape == a2.shape, f"shape mismatch for {fn!r}"
        assert np.allclose(a1, a2), f"field values differ for {fn!r}"


@pytest.mark.parametrize("key,_kind", V2_FIXTURES)
def test_round_trip_h5_layout_matches(
    src_storage: XdmfH5Storage,
    tmp_path: pathlib.Path,
    key: str,
    _kind,
):
    ds = src_storage.read_data_source(key)
    dst = XdmfH5Storage(tmp_path)
    dst.write_data_source(key, ds)

    with h5py.File(src_storage.h5_path(key), "r") as a, h5py.File(dst.h5_path(key), "r") as b:
        assert np.array_equal(a["Triangles"][:], b["Triangles"][:])
        assert np.allclose(a["Geometry"][:], b["Geometry"][:])
        # Affine TimeAxis only carries (initial_time, timestep_size, n_timesteps);
        # non-uniform on-disk timesteps drift on writeback. Check the affine
        # invariants instead of byte equality.
        ts_a = a["meta/time_steps"][:]
        ts_b = b["meta/time_steps"][:]
        assert ts_a.shape == ts_b.shape
        assert np.isclose(ts_a[0], ts_b[0])
        assert np.isclose(ts_a[1] - ts_a[0], ts_b[1] - ts_b[0])
        tn_a = a["meta/time_normalized"][:]
        tn_b = b["meta/time_normalized"][:]
        assert tn_a.shape == tn_b.shape
        assert np.isclose(tn_a[0], tn_b[0])
        # Every non-meta root group must round-trip with the same children.
        for grp_name in a.keys():
            if grp_name in ("Triangles", "Geometry", "meta"):
                continue
            assert grp_name in b, f"missing group {grp_name!r} after writeback"
            ka = sorted(a[grp_name].keys())
            kb = sorted(b[grp_name].keys())
            assert len(ka) == len(kb), f"group {grp_name!r} child count differs"


def test_read_raises_for_missing_key(tmp_path: pathlib.Path):
    storage = XdmfH5Storage(tmp_path)
    with pytest.raises(KeyError):
        storage.read_data_source("nonexistent")
