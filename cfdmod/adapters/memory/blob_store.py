"""In-RAM :class:`BlobStore` (issue #147).

A dict of ``bytes`` keyed by string. Pairs with
:class:`cfdmod.adapters.xdmf_h5.XdmfH5BlobStorage` to run templates
entirely in memory, and lets a consumer unit-test an object-store
pipeline without a real backend.
"""

from __future__ import annotations

__all__ = ["MemoryBlobStore"]

from typing import Iterable


class MemoryBlobStore:
    """Dict-backed :class:`~cfdmod.core.protocols.BlobStore`."""

    __slots__ = ("_blobs",)

    def __init__(self) -> None:
        self._blobs: dict[str, bytes] = {}

    def get_bytes(self, key: str) -> bytes:
        if key not in self._blobs:
            raise KeyError(f"MemoryBlobStore has no blob under key {key!r}")
        return self._blobs[key]

    def put_bytes(self, key: str, data: bytes) -> None:
        self._blobs[key] = bytes(data)

    def list_keys(self) -> Iterable[str]:
        return list(self._blobs.keys())

    def __contains__(self, key: str) -> bool:
        return key in self._blobs
