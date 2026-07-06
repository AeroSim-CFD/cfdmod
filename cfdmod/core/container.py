"""Generic container of :class:`DataSource` (or any pickleable value).

Lifts the ``HFPIAnalysisResults`` pattern from
``cfdmod.hfpi.handler`` into the core layer:

- a ``dict[K, V]`` keyed by a frozen Pydantic case-parameters object
  (``HFPICaseParameters`` is one example, but the container does not
  care);
- ``join_by(callback)`` partitions the container by a derived key
  (e.g. "by direction", "by recurrence period");
- ``filter_by(callback)`` returns a sub-container;
- ``map_values(pipeline, *, pool=None)`` runs a pipeline over every
  value, optionally in parallel via an injected :class:`Pool`.

Phase 6 of the v3 plan aliases ``HFPIAnalysisResults`` to
``Container[HFPICaseParameters, ResultType]`` and rewrites the
existing ``join_by_*`` helpers as one-line wrappers over
:meth:`join_by`.
"""

from __future__ import annotations

__all__ = ["Container"]

from typing import Any, Callable, Generic, Hashable, Iterator, TypeVar

from pydantic import BaseModel, ConfigDict, Field

from cfdmod.core.protocols import Pool

K = TypeVar("K", bound=Hashable)
V = TypeVar("V")
T = TypeVar("T", bound=Hashable)


class Container(BaseModel, Generic[K, V]):
    """Hashable-keyed map of values, with parallel fanout and partition.

    Frozen at the model level; the underlying dict is replaced by
    construction of a new container rather than mutated.
    """

    model_config = ConfigDict(frozen=True, arbitrary_types_allowed=True)

    items: dict[K, V] = Field(default_factory=dict)

    def __len__(self) -> int:
        return len(self.items)

    def __iter__(self) -> Iterator[K]:  # type: ignore[override]
        return iter(self.items)

    def __contains__(self, key: K) -> bool:
        return key in self.items

    def __getitem__(self, key: K) -> V:
        return self.items[key]

    def keys(self):
        return self.items.keys()

    def values(self):
        return self.items.values()

    # ----- Functional updates -------------------------------------------------

    def with_item(self, key: K, value: V) -> "Container[K, V]":
        new_items = dict(self.items)
        new_items[key] = value
        return self.__class__(items=new_items)

    def without_key(self, key: K) -> "Container[K, V]":
        new_items = {k: v for k, v in self.items.items() if k != key}
        return self.__class__(items=new_items)

    def merge(self, other: "Container[K, V]") -> "Container[K, V]":
        new_items = dict(self.items)
        new_items.update(other.items)
        return self.__class__(items=new_items)

    # ----- Partition / filter ------------------------------------------------

    def filter_by(self, predicate: Callable[[K], bool]) -> "Container[K, V]":
        """Return a sub-container of entries whose key satisfies ``predicate``."""
        new_items = {k: v for k, v in self.items.items() if predicate(k)}
        return self.__class__(items=new_items)

    def join_by(self, callback: Callable[[K], T]) -> dict[T, "Container[K, V]"]:
        """Partition by a derived key.

        Mirrors ``HFPIAnalysisResults.join_by``: for every entry, run
        ``callback(key)`` to derive a *partition* key, then collect
        entries sharing each partition key into their own container.
        """
        partitions: dict[T, dict[K, V]] = {}
        for k, v in self.items.items():
            partition_key = callback(k)
            partitions.setdefault(partition_key, {})[k] = v
        return {pk: self.__class__(items=pv) for pk, pv in partitions.items()}

    # ----- Map over values ---------------------------------------------------

    def map_values(
        self,
        func: Callable[[V], Any],
        *,
        pool: Pool | None = None,
    ) -> "Container[K, Any]":
        """Apply ``func`` to every value.

        If ``pool`` is supplied, fanout runs through ``pool.map`` and
        the entries' order is preserved by re-zipping with the keys.
        Without a pool the work runs sequentially in insertion order.
        """
        keys = list(self.items.keys())
        values = list(self.items.values())
        if pool is None:
            new_values = [func(v) for v in values]
        else:
            new_values = pool.map(func, values)
        return self.__class__(items=dict(zip(keys, new_values)))
