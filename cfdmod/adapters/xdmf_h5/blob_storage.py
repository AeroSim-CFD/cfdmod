"""Object-store-backed XDMF+H5 :class:`Storage` (issue #147).

:class:`XdmfH5BlobStorage` implements the same :class:`Storage` contract
as :class:`XdmfH5Storage`, but reads and writes the h5 (+ optional xdmf)
bytes through an injected :class:`~cfdmod.core.protocols.BlobStore`
instead of a local directory. A consumer supplies a thin ``BlobStore``
over their object store (S3, GCS, a DB blob column); cfdmod stays free of
any cloud SDK.

The implementation bridges through a temporary directory and reuses the
existing :class:`XdmfH5Storage` byte layout verbatim -- so the on-disk /
in-object format is identical, and ``run_template`` runs against object
storage with no other change. On read, all fields are materialised into
an in-RAM :class:`MemoryFieldStore` (the temp file does not outlive the
call), which is the right trade-off for an object-store round-trip; a
pure streaming path can be added later without changing this seam.
"""

from __future__ import annotations

__all__ = ["XdmfH5BlobStorage"]

import pathlib
import tempfile
from typing import Iterable

from cfdmod.adapters.memory import MemoryFieldStore
from cfdmod.adapters.xdmf_h5.storage import XdmfH5Storage
from cfdmod.core.data_source import DataSource
from cfdmod.core.errors import StorageKeyError
from cfdmod.core.protocols import BlobStore


class XdmfH5BlobStorage:
    """:class:`Storage` that persists the XDMF+H5 bytes via a ``BlobStore``.

    Args:
        blobs: The blob backend. Blob keys are ``"<key>.h5"`` and, when
            written, ``"<key>.xdmf"``.
        write_xdmf: When True, ``write_data_source`` also stores the
            ``.xdmf`` sidecar blob. Default True.
    """

    __slots__ = ("_blobs", "_write_xdmf")

    def __init__(self, blobs: BlobStore, *, write_xdmf: bool = True) -> None:
        self._blobs = blobs
        self._write_xdmf = bool(write_xdmf)

    def __contains__(self, key: str) -> bool:
        return f"{key}.h5" in self._blobs

    def keys(self) -> Iterable[str]:
        return sorted(k[: -len(".h5")] for k in self._blobs.list_keys() if k.endswith(".h5"))

    def read_data_source(self, key: str) -> DataSource:
        blob_key = f"{key}.h5"
        try:
            data = self._blobs.get_bytes(blob_key)
        except KeyError as exc:
            raise StorageKeyError(
                f"XdmfH5BlobStorage has no data source under key {key!r} (blob {blob_key!r})"
            ) from exc

        with tempfile.TemporaryDirectory() as td:
            root = pathlib.Path(td)
            local_path = root / f"{key}.h5"
            local_path.parent.mkdir(parents=True, exist_ok=True)
            local_path.write_bytes(data)
            ds = XdmfH5Storage(root).read_data_source(key)
            # Materialise fields into RAM so the DataSource does not outlive
            # the temp file it was lazily reading from.
            arrays = {name: ds.fields.read(name) for name in ds.fields.keys()}
            return ds.model_copy(update={"fields": MemoryFieldStore(arrays)})

    def write_data_source(self, key: str, ds: DataSource) -> None:
        with tempfile.TemporaryDirectory() as td:
            root = pathlib.Path(td)
            XdmfH5Storage(root, write_xdmf=self._write_xdmf).write_data_source(key, ds)
            for suffix in (".h5", ".xdmf"):
                produced = root / f"{key}{suffix}"
                if produced.exists():
                    self._blobs.put_bytes(f"{key}{suffix}", produced.read_bytes())

    # --- Freshness --------------------------------------------------------

    def digest(self, key: str, strategy: str = "size_mtime") -> str:
        """Change-detecting token for the object bytes under ``key``.

        A plain ``BlobStore`` exposes no native size/mtime/ETag, so every
        strategy degrades to a content hash of the ``.h5`` (+ ``.xdmf``)
        blobs. The requested strategy is embedded so a strategy switch is
        still a change signal. A backend that exposes an S3 ETag / checksum
        can subclass and override ``backend`` without transferring bytes.
        """
        import hashlib

        blob_key = f"{key}.h5"
        try:
            data = self._blobs.get_bytes(blob_key)
        except KeyError as exc:
            raise StorageKeyError(
                f"XdmfH5BlobStorage has no data source under key {key!r} (blob {blob_key!r})"
            ) from exc
        h = hashlib.blake2b(digest_size=32)
        h.update(data)
        xdmf_key = f"{key}.xdmf"
        if xdmf_key in self._blobs:
            h.update(self._blobs.get_bytes(xdmf_key))
        return f"{strategy}:blob:{h.hexdigest()}"

    def read_signature(self, key: str) -> str | None:
        sig_key = f"{key}.sig"
        if sig_key not in self._blobs:
            return None
        return self._blobs.get_bytes(sig_key).decode("utf-8")

    def write_signature(self, key: str, signature: str) -> None:
        self._blobs.put_bytes(f"{key}.sig", signature.encode("utf-8"))
