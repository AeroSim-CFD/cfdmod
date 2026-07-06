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

import importlib
from typing import Any

__all__ = [
    "MemoryFieldStore",
    "MemoryStorage",
    "H5FieldStore",
    "XdmfH5Storage",
]

# Lazy so importing the light in-RAM adapter (which the ops layer does) does
# not drag in the h5py-backed xdmf_h5 adapter and, through it, cfdmod.io
# (pandas / pyarrow). See issue #147.
_SYMBOL_MODULE = {
    "MemoryFieldStore": "cfdmod.adapters.memory",
    "MemoryStorage": "cfdmod.adapters.memory",
    "H5FieldStore": "cfdmod.adapters.xdmf_h5",
    "XdmfH5Storage": "cfdmod.adapters.xdmf_h5",
}


def __getattr__(name: str) -> Any:
    module_path = _SYMBOL_MODULE.get(name)
    if module_path is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    value = getattr(importlib.import_module(module_path), name)
    globals()[name] = value
    return value


def __dir__() -> list[str]:
    return sorted(set(__all__) | set(globals()))
