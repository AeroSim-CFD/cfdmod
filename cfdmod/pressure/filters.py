"""Deprecated re-exports.

This module has moved to :mod:`cfdmod.filters` so the filter pipeline
can be used independently of the pressure-coefficient pipeline. Update
imports::

    # old
    from cfdmod.pressure.filters import apply_filters, MovingAverageFilter

    # new
    from cfdmod.filters import (
        apply_filters,           # pure-numpy core
        apply_filters_h5,        # file-in / file-out wrapper
        MovingAverageFilter,
    )

The old ``apply_filters(input_h5, output_h5, ...)`` (file-in / file-out)
is now :func:`cfdmod.filters.apply_filters_h5`. The new
:func:`cfdmod.filters.apply_filters` is the pure-numpy core. This
shim re-exports the file-based version under the old name for
backward compatibility, with a ``DeprecationWarning``.
"""

from __future__ import annotations

import warnings

from cfdmod.filters import FilterSpec, MovingAverageFilter
from cfdmod.filters import apply_filters_h5 as _apply_filters_h5
from cfdmod.filters.core import _window_in_samples

__all__ = ["MovingAverageFilter", "FilterSpec", "apply_filters", "_window_in_samples"]

warnings.warn(
    "cfdmod.pressure.filters has moved to cfdmod.filters. The H5 wrapper is "
    "now cfdmod.filters.apply_filters_h5; cfdmod.filters.apply_filters is the "
    "pure-numpy core. This module will be removed in a future release.",
    DeprecationWarning,
    stacklevel=2,
)

# Backward-compat name: the old top-level apply_filters was the file flow.
apply_filters = _apply_filters_h5
