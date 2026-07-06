"""In-RAM :class:`Storage`.

Keeps full :class:`DataSource` objects in a dict, keyed by string.
Pure metadata + arrays; no serialisation. Used by every unit test and
by recipe-shape tests that do not need a real file flow.
"""

from __future__ import annotations

__all__ = ["MemoryStorage"]

from typing import Iterable

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

    __slots__ = ("_items",)

    def __init__(self) -> None:
        self._items: dict[str, DataSource] = {}

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
