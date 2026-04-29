# Backward compatibility shim. Import from cfdmod.config and cfdmod.io instead.
import warnings

warnings.warn(
    "cfdmod.api is a deprecated compatibility shim and will be removed in a "
    "future release. Import from cfdmod.config and cfdmod.io instead.",
    DeprecationWarning,
    stacklevel=2,
)

__all__ = [
    "HashableConfig",
    "read_stl",
    "export_stl",
]

from cfdmod.config import HashableConfig
from cfdmod.io import read_stl, export_stl
