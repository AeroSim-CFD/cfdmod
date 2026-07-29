"""The in-RAM content digest must not cost two copies of the data.

``MemoryStorage.digest`` hashes the topology and every field. It did that by
building ``np.ascontiguousarray(arr, dtype=float64).tobytes()`` in one go -- an
upcast copy plus a bytes copy -- so digesting a 64 MB float32 field peaked at
256 MB. That runs on every ``run_template`` with declared outputs, which made it
the dominant cost of a run whose whole point was bounding memory.

Two properties matter, and the first one is why the fix is safe: the byte
sequence fed to the hash is unchanged, so signatures written before it still
match.
"""

from __future__ import annotations

import hashlib
import tracemalloc

import numpy as np
import pytest

from cfdmod.adapters.memory import MemoryFieldStore, MemoryStorage
from cfdmod.adapters.memory.storage import _update_blockwise
from cfdmod.core import ElementMeta, SurfaceDataSource, TimeAxis, Topology

pytestmark = pytest.mark.unit


def _whole_array_hash(arr: np.ndarray, dtype) -> str:
    """The pre-fix formulation, kept here as the reference to match."""
    h = hashlib.blake2b(digest_size=32)
    h.update(np.ascontiguousarray(arr, dtype=dtype).tobytes())
    return h.hexdigest()


def _blockwise_hash(arr: np.ndarray, dtype) -> str:
    h = hashlib.blake2b(digest_size=32)
    _update_blockwise(h, arr, dtype)
    return h.hexdigest()


@pytest.mark.parametrize(
    "shape",
    [
        (5000, 300),  # many rows: actually blocks
        (1, 10),  # single row
        (10, 1),  # single column
        (77, 3, 4),  # 3-D, e.g. vertices
        (0, 5),  # empty
        (13,),  # 1-D, time-aggregated field
    ],
)
@pytest.mark.parametrize("dtype", [np.float64, np.int64])
def test_blockwise_hash_is_byte_identical(shape, dtype):
    """Same bytes in, same digest out -- for every shape, including degenerate."""
    arr = (np.random.default_rng(0).random(shape) * 100).astype(np.float32)
    assert _blockwise_hash(arr, dtype) == _whole_array_hash(arr, dtype)


def test_blockwise_hash_still_distinguishes_content():
    """A hash that bounded memory by hashing less would pass the test above."""
    a = np.arange(6000, dtype=np.float32).reshape(2000, 3)
    b = a.copy()
    b[-1, -1] += 1.0  # last element, i.e. in the final block
    assert _blockwise_hash(a, np.float64) != _blockwise_hash(b, np.float64)


def test_digest_peak_is_bounded_not_proportional_to_the_field():
    """Measured, not argued: the digest must not scale with field size."""
    n_elements, n_timesteps = 20_000, 400
    rng = np.random.default_rng(1)
    verts = rng.random((n_elements * 3, 3))
    tris = np.arange(n_elements * 3, dtype=np.int32).reshape(n_elements, 3)
    field_bytes = n_elements * n_timesteps * 4
    ds = SurfaceDataSource(
        time=TimeAxis(initial_time=0.0, timestep_size=0.1, n_timesteps=n_timesteps),
        topology=Topology.triangles(tris, verts),
        elements=ElementMeta(),
        fields=MemoryFieldStore(
            {"pressure": rng.random((n_elements, n_timesteps)).astype(np.float32)}
        ),
    )
    storage = MemoryStorage()
    storage.write_data_source("body", ds)

    tracemalloc.start()
    try:
        storage.digest("body")
        peak = tracemalloc.get_traced_memory()[1]
    finally:
        tracemalloc.stop()

    # Pre-fix this was ~4x the field (float64 upcast + tobytes, both whole).
    assert peak < field_bytes, (
        f"digest peaked at {peak / 1e6:.1f} MB for a {field_bytes / 1e6:.1f} MB field"
    )


def test_digest_still_changes_when_a_field_changes():
    """The end-to-end property freshness depends on."""
    verts = np.array([[0, 0, 0], [1, 0, 0], [1, 1, 0]], dtype=np.float64)
    tris = np.array([[0, 1, 2]], dtype=np.int32)

    def source(value: float) -> SurfaceDataSource:
        return SurfaceDataSource(
            time=TimeAxis(initial_time=0.0, timestep_size=0.1, n_timesteps=2),
            topology=Topology.triangles(tris, verts),
            elements=ElementMeta(),
            fields=MemoryFieldStore({"cp": np.array([[value, 2.0]])}),
        )

    storage = MemoryStorage()
    storage.write_data_source("a", source(1.0))
    storage.write_data_source("b", source(1.0))
    storage.write_data_source("c", source(1.5))

    assert storage.digest("a") == storage.digest("b")
    assert storage.digest("a") != storage.digest("c")
