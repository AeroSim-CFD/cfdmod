"""Deprecated re-exports.

The streaming H5 statistics flow has moved to :mod:`cfdmod.statistics`
so it can be used independently of the pressure-coefficient pipeline.
Update imports::

    # old
    from cfdmod.pressure.statistics_runner import calculate_statistics_from_h5

    # new
    from cfdmod.statistics import (
        apply_statistics,         # pure-numpy core
        apply_statistics_h5,      # file-in wrapper
    )

This shim re-exports the H5 wrapper under the old name with a
``DeprecationWarning``. The pure-numpy entry point
:func:`cfdmod.statistics.apply_statistics` is new and has no analogue
under the old layout.
"""

from __future__ import annotations

import warnings

from cfdmod.statistics import apply_statistics_h5

__all__ = ["calculate_statistics_from_h5"]

warnings.warn(
    "cfdmod.pressure.statistics_runner has moved to cfdmod.statistics. "
    "Use cfdmod.statistics.apply_statistics_h5 (file flow) or "
    "cfdmod.statistics.apply_statistics (numpy core). This module will be "
    "removed in a future release.",
    DeprecationWarning,
    stacklevel=2,
)

# Backward-compat name.
calculate_statistics_from_h5 = apply_statistics_h5
