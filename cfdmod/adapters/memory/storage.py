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

    def read_data_source(self, key: str) -> DataSource:
        if key not in self._items:
            raise StorageKeyError(f"MemoryStorage has no data source under key {key!r}")
        return self._items[key]

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


def _data_source_content_hash(ds: DataSource) -> str:
    """Stable hash of a data source's kind, topology, time axis, and fields."""
    h = hashlib.blake2b(digest_size=32)
    h.update(ds.kind.encode("utf-8"))
    if ds.topology is not None:
        h.update(np.ascontiguousarray(ds.topology.vertices, dtype=np.float64).tobytes())
        conn = ds.topology.connectivity
        if conn is not None:
            h.update(np.ascontiguousarray(conn, dtype=np.int64).tobytes())
    h.update(str(ds.time.n_timesteps).encode("utf-8"))
    for name in sorted(ds.fields.keys()):
        arr = np.ascontiguousarray(ds.fields.read(name), dtype=np.float64)
        h.update(name.encode("utf-8"))
        h.update(str(arr.shape).encode("utf-8"))
        h.update(arr.tobytes())
    return h.hexdigest()
