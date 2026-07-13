"""Default numeric precision for field-**value** arrays.

Field values -- pressure, Cp, per-element force/moment and their time series --
are cheap to keep in **float32**: it is enough for these quantities and halves
their memory on top of the time-chunking (:mod:`cfdmod.core.chunked`). The raw
solver export is float32 on disk anyway.

Design (the "float32 sweep"):

- The field ops **preserve their input field's dtype** instead of upcasting to
  float64 (they used to hard-cast). So float32 data flows as float32 end to end,
  while a float64 source stays float64 (exactness / precision unchanged) --
  ``force_contribution`` / ``moment_contribution`` infer the dtype from the field
  and cast the geometric factors to match.
- :data:`FIELD_DTYPE` is the default precision the **load path**
  (:func:`cfdmod.building.per_floor_loads`) casts its source pressure to, so the
  whole per-triangle chain runs in float32 by default. Change it to shift that
  default library-wide.

What stays float64 regardless (precision, negligible memory): mesh geometry /
coordinates, the structural-model definition and the modal / dynamics solve, the
time axis, and the on-disk storage format (``XdmfH5Storage`` normalises field
data to float64 on write, so round-trips are unchanged).
"""

from __future__ import annotations

__all__ = ["FIELD_DTYPE", "as_field"]

import numpy as np

FIELD_DTYPE = np.dtype("float32")


def as_field(arr: np.ndarray) -> np.ndarray:
    """``np.asarray(arr, dtype=FIELD_DTYPE)`` -- cast a field-value array to the
    default field precision."""
    return np.asarray(arr, dtype=FIELD_DTYPE)
