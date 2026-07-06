"""Object-store-backed Storage adapter (issue #147).

XdmfH5BlobStorage must round-trip a DataSource through a BlobStore with
byte-for-byte the same h5 payload as the on-disk XdmfH5Storage, and be a
drop-in Storage for run_template.
"""

from __future__ import annotations

import numpy as np
import pytest

from cfdmod.adapters.memory import MemoryBlobStore, MemoryFieldStore
from cfdmod.adapters.xdmf_h5 import XdmfH5BlobStorage, XdmfH5Storage
from cfdmod.core import (
    ElementMeta,
    PipelineTemplate,
    StorageKeyError,
    SurfaceDataSource,
    TimeAxis,
    Topology,
    run_template,
)

pytestmark = pytest.mark.unit


def _surface(n_t: int = 6) -> SurfaceDataSource:
    verts = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0], [1, 1, 0]], dtype=np.float64)
    tris = np.array([[0, 1, 2], [1, 3, 2]], dtype=np.int32)
    rng = np.random.default_rng(0)
    return SurfaceDataSource(
        time=TimeAxis(initial_time=0.0, timestep_size=0.1, n_timesteps=n_t),
        topology=Topology.triangles(tris, verts),
        elements=ElementMeta(),
        fields=MemoryFieldStore({"pressure": rng.normal(size=(2, n_t))}),
    )


def test_round_trip_matches_on_disk_bytes(tmp_path):
    ds = _surface()

    blobs = MemoryBlobStore()
    XdmfH5BlobStorage(blobs).write_data_source("out/cp", ds)

    # Same source written to a real directory must yield identical h5 bytes.
    XdmfH5Storage(tmp_path).write_data_source("out/cp", ds)
    on_disk = (tmp_path / "out/cp.h5").read_bytes()
    assert blobs.get_bytes("out/cp.h5") == on_disk
    assert "out/cp.xdmf" in blobs  # sidecar stored too


def test_round_trip_values(tmp_path):
    ds = _surface()
    blobs = MemoryBlobStore()
    storage = XdmfH5BlobStorage(blobs)
    storage.write_data_source("cp", ds)

    back = storage.read_data_source("cp")
    assert back.kind == "surface"
    np.testing.assert_allclose(back.fields.read("pressure"), ds.fields.read("pressure"), rtol=1e-9)
    assert set(storage.keys()) == {"cp"}


def test_missing_key_raises_storage_key_error():
    with pytest.raises(StorageKeyError):
        XdmfH5BlobStorage(MemoryBlobStore()).read_data_source("nope")


def test_run_template_against_blob_storage():
    blobs = MemoryBlobStore()
    XdmfH5BlobStorage(blobs).write_data_source("body", _surface())

    tpl = PipelineTemplate(
        name="cp_stats",
        root="",
        inputs={"body": {"kind": "surface", "path": "body"}},
        pipeline=[
            {
                "id": "scaled",
                "kind": "scale",
                "source": "body",
                "field": "pressure",
                "factor": 2.0,
            },
            {
                "id": "stats",
                "kind": "statistics",
                "source": "scaled",
                "field": "pressure",
                "kinds": ["mean", "rms"],
            },
        ],
        outputs={"stats": {"source": "stats", "path": "cp.stats"}},
    )
    run_template(tpl, storage=XdmfH5BlobStorage(blobs))

    # The stats output was persisted back to the blob store and reloads.
    stats = XdmfH5BlobStorage(blobs).read_data_source("cp.stats")
    assert stats.time.is_time_aggregated
    assert "cp.stats.h5" in blobs
