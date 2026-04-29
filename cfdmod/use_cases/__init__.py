# Backward compatibility shim. Import from top-level domain modules instead.
# e.g.: from cfdmod.loft import LoftParams  instead of  from cfdmod.use_cases.loft import LoftParams
import warnings

warnings.warn(
    "cfdmod.use_cases is a deprecated compatibility shim and will be removed in "
    "a future release. Import from the top-level domain modules instead "
    "(cfdmod.loft, cfdmod.pressure, cfdmod.roughness, ...).",
    DeprecationWarning,
    stacklevel=2,
)

from cfdmod import loft, roughness, pressure, s1, climate, analytical, snapshot, altimetry, hfpi
