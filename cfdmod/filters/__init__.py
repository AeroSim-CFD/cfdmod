"""Signal-processing filters for cfdmod.

A filter is a 1-D operation applied independently to each column of a
2D array (or to a 1D signal). Filters are first-class pipeline steps
and not coupled to any specific coefficient: the same chain that
smooths a Cp series can also smooth any other timeseries the caller
supplies.

Two entry points:

- :func:`apply_filters` -- pure numpy. Pass a ``(n_time,)`` or
  ``(n_time, n_features)`` array, the sample spacing ``dt``, and a
  list of filter specs. Use this from notebooks, custom pipelines, or
  any source that is not cfdmod's standard H5 layout.
- :func:`apply_filters_h5` -- file-in / file-out wrapper around the
  numpy core. Reads cfdmod's H5 layout, runs the chain, writes the
  filtered series + sibling XDMF + ``processing_metadata``. Use this
  for the standard end-to-end pipeline.

Adding a new filter type:

1. Add a new Pydantic model with a unique ``kind`` Literal in
   :mod:`cfdmod.filters.specs` and register it in the ``FilterSpec``
   union.
2. Add a dispatch branch in :func:`cfdmod.filters.core._apply_one`.
"""

from cfdmod.filters.core import apply_filters
from cfdmod.filters.h5 import apply_filters_h5
from cfdmod.filters.specs import FilterSpec, MovingAverageFilter

__all__ = [
    "MovingAverageFilter",
    "FilterSpec",
    "apply_filters",
    "apply_filters_h5",
]
