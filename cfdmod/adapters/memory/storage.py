"""In-RAM :class:`Storage`.

Keeps full :class:`DataSource` objects in a dict, keyed by string.
Pure metadata + arrays; no serialisation. Used by every unit test and
by recipe-shape tests that do not need a real file flow.
"""

from __future__ import annotations

__all__ = ["MemoryStorage"]

import hashlib
from typing import Iterable

import numpy as np

from cfdmod.core.data_source import DataSource
from cfdmod.core.errors import StorageKeyError


class MemoryStorage:
    """Dict-backed :class:`Storage`.

    Stores complete :class:`DataSource` objects in a Python dict.
    ``read_data_source`` and ``write_data_source`` are O(1) hash
    lookups; nothing is copied.

    A :class:`MemoryStorage` is *mutable*: new keys are added by
    :meth:`write_data_source`. The data sources themselves remain
    frozen, so this is consistent with the functional-core principle.
    """

    __slots__ = ("_items", "_signatures")

    def __init__(self) -> None:
        self._items: dict[str, DataSource] = {}
        self._signatures: dict[str, str] = {}

    def keys(self) -> Iterable[str]:
        return self._items.keys()

    def __contains__(self, key: str) -> bool:
        return key in self._items

    def read_data_source(self, key: str, *, kind: str | None = None) -> DataSource:
        if key not in self._items:
            raise StorageKeyError(f"MemoryStorage has no data source under key {key!r}")
        ds = self._items[key]
        # RAM keeps the concrete type, so there is nothing to infer -- but the
        # check still has to exist, or a caller passing ``kind`` would get a
        # silent pass here and a hard failure against the h5 backend.
        if kind is not None and ds.kind != kind:
            raise ValueError(
                f"MemoryStorage holds a {ds.kind!r} data source under key {key!r}, "
                f"but {kind!r} was requested"
            )
        return ds

    def write_data_source(self, key: str, ds: DataSource) -> None:
        self._items[key] = ds

    # --- Freshness --------------------------------------------------------

    def digest(self, key: str, strategy: str = "size_mtime") -> str:
        """Content digest of the in-RAM data source under ``key``.

        RAM has no size/mtime, so every strategy degrades to a stable
        content hash of the topology + fields. The requested strategy is
        still embedded so a signature computed with one strategy does not
        silently match one computed with another.
        """
        if key not in self._items:
            raise StorageKeyError(f"MemoryStorage has no data source under key {key!r}")
        return f"{strategy}:mem:{_data_source_content_hash(self._items[key])}"

    def read_signature(self, key: str) -> str | None:
        return self._signatures.get(key)

    def write_signature(self, key: str, signature: str) -> None:
        self._signatures[key] = signature


# Rows of a field hashed at a time. Bounds the float64 upcast + tobytes copy
# to roughly this many bytes instead of the whole field twice over.
_HASH_BLOCK_BYTES = 8 << 20


def _update_blockwise(h, arr: np.ndarray, dtype) -> None:
    """Feed ``arr`` to ``h`` as ``dtype`` bytes, a row block at a time.

    Hashing ``np.ascontiguousarray(arr, dtype).tobytes()`` in one go costs two
    full copies of the array -- the upcast and the bytes -- so digesting a
    64 MB float32 field peaked at 256 MB. Since the array is walked in
    C order, concatenating row blocks yields exactly the same byte sequence,
    so this is byte-identical to the whole-array form and existing signatures
    keep matching. Only the peak changes.
    """
    arr = np.asarray(arr)
    if arr.size == 0:
        return
    itemsize = np.dtype(dtype).itemsize
    row_bytes = max(1, int(np.prod(arr.shape[1:], dtype=np.int64))) * itemsize
    rows_per_block = max(1, _HASH_BLOCK_BYTES // max(1, row_bytes))
    for start in range(0, arr.shape[0], rows_per_block):
        block = np.ascontiguousarray(arr[start : start + rows_per_block], dtype=dtype)
        h.update(block.tobytes())


def _data_source_content_hash(ds: DataSource) -> str:
    """Stable hash of a data source's kind, topology, time axis, and fields."""
    h = hashlib.blake2b(digest_size=32)
    h.update(ds.kind.encode("utf-8"))
    if ds.topology is not None:
        _update_blockwise(h, ds.topology.vertices, np.float64)
        conn = ds.topology.connectivity
        if conn is not None:
            _update_blockwise(h, conn, np.int64)
    h.update(str(ds.time.n_timesteps).encode("utf-8"))
    for name in sorted(ds.fields.keys()):
        arr = np.asarray(ds.fields.read(name))
        h.update(name.encode("utf-8"))
        h.update(str(arr.shape).encode("utf-8"))
        _update_blockwise(h, arr, np.float64)
    return h.hexdigest()
