"""Window-selection time op.

Restrict a :class:`DataSource` to a ``[start, end]`` time interval.
The op mutates the affine :class:`TimeAxis` and slices every field
along its time axis (no copy on the memory adapter when the slice is
contiguous).
"""

from __future__ import annotations

__all__ = ["WindowSelectionParams", "window_selection"]

from typing import ClassVar, Literal

from cfdmod.core.data_source import DataSource
from cfdmod.core.ops import OpParams


class WindowSelectionParams(OpParams):
    """Parameters for :func:`window_selection`.

    Attributes:
        start: Window start, in the same time units as ``ds.time``.
        end: Window end (inclusive of the nearest sample).
    """

    kind: Literal["time_window"] = "time_window"
    start: float
    end: float

    chunkable_along: ClassVar[frozenset[str]] = frozenset({"time", "elements"})


def window_selection(ds: DataSource, p: WindowSelectionParams) -> DataSource:
    new_time, idx_slice = ds.time.window(p.start, p.end)

    new_store = ds.fields
    for name in list(ds.fields.keys()):
        arr = ds.fields.read(name, time_slice=idx_slice)
        new_store = new_store.with_field(name, arr)

    return ds.model_copy(update={"time": new_time, "fields": new_store})
