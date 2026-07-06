"""Concrete backends for the v3 core protocols.

Two adapters land in Phase 1:

- ``cfdmod.adapters.memory`` -- in-RAM numpy backend. ``MemoryFieldStore``
  is ~50 lines; ``MemoryStorage`` is a dict keyed by string. Used by
  every test and by notebooks doing exploratory work.
- ``cfdmod.adapters.xdmf_h5`` -- wraps the existing
  ``cfdmod.io.xdmf`` writers. ``H5FieldStore`` does h5py slab reads on
  demand; ``XdmfH5Storage`` resolves a logical key to an
  ``<key>.h5`` + ``<key>.xdmf`` pair under its root directory. The
  on-disk format is unchanged.

The core package never imports adapters. The shell (recipe runners,
CLIs) wires them in by constructing a :class:`Context`.
"""

from __future__ import annotations

from cfdmod.adapters.memory import MemoryFieldStore, MemoryStorage
from cfdmod.adapters.xdmf_h5 import H5FieldStore, XdmfH5Storage

__all__ = [
    "MemoryFieldStore",
    "MemoryStorage",
    "H5FieldStore",
    "XdmfH5Storage",
]
