"""Protocols (DI seams) used by the v3 core abstractions.

These are the *only* abstractions the core package depends on at
runtime. Concrete backends -- in-RAM numpy, h5py + XDMF, anything else
in the future -- live under ``cfdmod/adapters/`` and implement these
protocols.

Why protocols and not abstract base classes: protocols are structural,
so an adapter does not need to inherit from a class to satisfy the
contract. Tests can pass plain classes or even dataclasses. Adapters
remain easy to swap.

Three protocols matter most:

- :class:`FieldStore` -- the single seam between the small-data
  (numpy in RAM) and large-data (h5 file on disk) paths. Every op
  reads / writes fields exclusively through this protocol; the op
  layer never imports ``h5py``.
- :class:`Storage` -- whole-:class:`DataSource` round-trips. Pairs
  with one ``FieldStore`` per data source.
- :class:`Logger` and :class:`Pool` -- bundled in the shell
  ``Context`` and passed explicitly to recipes. Default to the
  no-op stubs in the memory adapter when not supplied.
"""

from __future__ import annotations

__all__ = [
    "FieldStore",
    "Storage",
    "BlobStore",
    "Logger",
    "Pool",
]

from typing import (
    TYPE_CHECKING,
    Any,
    Callable,
    Iterable,
    Protocol,
    runtime_checkable,
)

if TYPE_CHECKING:
    import numpy as np

    from cfdmod.core.data_source import DataSource


@runtime_checkable
class FieldStore(Protocol):
    """Per-:class:`DataSource` field store.

    A field store maps field names (``"pressure"``, ``"ux"``, ...) to
    arrays of shape ``(n_elements, n_timesteps)`` or ``(n_elements,)``
    for time-aggregated outputs. The adapter is free to decide whether
    those arrays live in RAM (:class:`MemoryFieldStore`) or are read on
    demand from an h5 file (:class:`H5FieldStore`).

    Slicing is the lazy-read mechanism. ``read(name, time_slice=slice(0,
    1024))`` materialises only that 2D slab. Adapters that hold the
    whole array in memory still honour the slice -- they just slice the
    in-RAM array.

    All updates are functional: :meth:`with_field` returns a new
    ``FieldStore`` rather than mutating in place. The memory adapter
    shares unmodified arrays by reference; the h5 adapter writes
    through to disk under ``r+`` mode and otherwise overlays a
    :class:`MemoryFieldStore` on top.
    """

    def keys(self) -> Iterable[str]:
        """Iterate the names of every field carried by this store."""
        ...

    def shape(self, name: str) -> tuple[int, ...]:
        """Shape of the named field.

        Either ``(n_elements, n_timesteps)`` for time-resolved fields or
        ``(n_elements,)`` for time-aggregated fields (statistics, single
        snapshots).
        """
        ...

    def dtype(self, name: str) -> Any:
        """``numpy`` dtype of the named field."""
        ...

    def read(
        self,
        name: str,
        *,
        time_slice: slice | None = None,
        element_slice: slice | None = None,
        elements: "np.ndarray | None" = None,
    ) -> "np.ndarray":
        """Read a slab of the named field.

        Args:
            name: Field name (must be in :meth:`keys`).
            time_slice: Optional slice along the time axis. ``None`` ->
                full time range.
            element_slice: Optional slice along the elements axis.
                ``None`` -> full elements range.
            elements: Optional fancy-indexed element array. Mutually
                exclusive with ``element_slice``.

        Returns:
            A 1-D or 2-D ``numpy.ndarray`` corresponding to the
            requested slab. The first axis is always elements.
        """
        ...

    def write(
        self,
        name: str,
        value: "np.ndarray",
        *,
        time_slice: slice | None = None,
        element_slice: slice | None = None,
    ) -> None:
        """Write a slab of the named field. In-place on disk-backed
        stores; raises on read-only stores."""
        ...

    def with_field(self, name: str, value: "np.ndarray") -> "FieldStore":
        """Return a new ``FieldStore`` with ``name`` bound to ``value``.

        Functional update: the original store is unchanged. Adapters
        are free to share the array by reference rather than copying.
        """
        ...


@runtime_checkable
class Storage(Protocol):
    """Whole-:class:`DataSource` round-trip backend.

    A storage handles the topology + element metadata + groupings + time
    axis, and hands out the matching :class:`FieldStore` for the
    associated field arrays. Recipes take a ``Storage`` and a logical
    key; tests pass :class:`MemoryStorage`, production CLIs pass
    :class:`XdmfH5Storage`. No code path differs between the two.
    """

    def read_data_source(self, key: str) -> "DataSource":
        """Read a complete :class:`DataSource` by logical key.

        ``key`` is a string identifier. The memory backend treats it
        as a dict key; the h5 backend resolves it to an
        ``<key>.h5`` + ``<key>.xdmf`` pair under its root directory.
        """
        ...

    def write_data_source(self, key: str, ds: "DataSource") -> None:
        """Write a complete :class:`DataSource` under ``key``."""
        ...

    def keys(self) -> Iterable[str]:
        """Iterate the logical keys held by this storage."""
        ...

    # --- Freshness / provenance (optional) --------------------------------
    #
    # The three methods below back output-staleness detection (skip vs.
    # recompute). They are part of the protocol so every built-in adapter
    # implements them, but the default ``run_template`` path calls them only
    # when the backend advertises them (``hasattr``), so a pre-existing
    # third-party ``Storage`` that omits them keeps working unchanged.

    def digest(self, key: str, strategy: str = "size_mtime") -> str:
        """Return a cheap change-detecting token for the object under ``key``.

        ``strategy`` selects how the token is derived:

        - ``"size_mtime"`` -- size + mtime, no byte reads (default; fast).
        - ``"content"`` -- a strong content hash of the stored bytes.
        - ``"backend"`` -- the backend's own token (e.g. an object-store
          ETag) resolved without transferring bytes; adapters with no
          native token fall back to ``size_mtime``.

        The returned string embeds the strategy so switching strategy is
        itself a change signal. Raises :class:`StorageKeyError` when the
        key is absent.
        """
        ...

    def read_signature(self, key: str) -> "str | None":
        """Return the freshness signature stamped on ``key``, or ``None``.

        ``None`` means "no signature recorded" (never written with
        freshness tracking) and is treated as ``missing`` by the caller.
        """
        ...

    def write_signature(self, key: str, signature: str) -> None:
        """Stamp ``signature`` onto an already-written object at ``key``."""
        ...


@runtime_checkable
class BlobStore(Protocol):
    """A flat key -> bytes blob backend.

    The seam between cfdmod's on-disk XDMF+H5 byte layout and a
    non-filesystem backing store (an object store such as S3, a database
    blob column, an in-RAM dict). A consumer implements these four
    methods over their client of choice; cfdmod stays free of any cloud
    SDK. :class:`cfdmod.adapters.xdmf_h5.XdmfH5BlobStorage` pairs a
    ``BlobStore`` with the existing XDMF+H5 reader/writer so a template
    can run against object storage with no other change.

    Keys are the full object names *including* extension
    (``"out/cp.time_series.h5"``), so the ``.h5`` and its optional
    ``.xdmf`` sidecar are distinct blobs.
    """

    def get_bytes(self, key: str) -> bytes:
        """Return the bytes stored under ``key``; raise ``KeyError`` if absent."""
        ...

    def put_bytes(self, key: str, data: bytes) -> None:
        """Store ``data`` under ``key``, overwriting any existing blob."""
        ...

    def list_keys(self) -> Iterable[str]:
        """Iterate every blob key currently held."""
        ...

    def __contains__(self, key: str) -> bool:
        """Whether a blob exists under ``key``."""
        ...


@runtime_checkable
class Logger(Protocol):
    """Minimal structured-log seam used by the shell.

    The core package never imports ``loguru`` directly; instead it
    receives a :class:`Logger` via :class:`Context`. Tests pass a no-op
    or a list-collector for assertions. Production CLIs adapt
    ``loguru`` into this protocol.
    """

    def info(self, message: str, /, **fields: Any) -> None: ...
    def warning(self, message: str, /, **fields: Any) -> None: ...
    def error(self, message: str, /, **fields: Any) -> None: ...
    def debug(self, message: str, /, **fields: Any) -> None: ...


@runtime_checkable
class Pool(Protocol):
    """Parallel-fanout seam for :class:`Container.map_values`.

    Mirrors the shape of :class:`multiprocessing.pool.Pool` enough to
    swap a real pool, a thread pool, or a dummy synchronous pool. Pure
    sequential runs simply leave the pool as ``None``.
    """

    def map(self, func: Callable[..., Any], iterable: Iterable[Any]) -> list[Any]: ...
