"""XDMF + H5 adapter for the v3 core protocols.

Wraps the existing ``cfdmod.io.xdmf`` writers without changing the
on-disk format. The :class:`XdmfH5Storage` writes to disk *exactly*
the same layout shape that the v2 pressure pipeline produces today;
this is the property phase 2 round-trip tests assert on every
fixture.
"""

from cfdmod.adapters.xdmf_h5.field_store import H5FieldStore
from cfdmod.adapters.xdmf_h5.storage import XdmfH5Storage

__all__ = ["H5FieldStore", "XdmfH5Storage"]
