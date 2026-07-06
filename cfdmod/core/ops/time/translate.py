"""Time-translation op.

Sets ``initial_time`` to a new value, leaving the timestep size and
length untouched. Pure metadata update; field arrays are not read.
"""

from __future__ import annotations

__all__ = ["TranslateParams", "translate"]

from typing import ClassVar, Literal

from cfdmod.core.data_source import DataSource
from cfdmod.core.ops import OpParams


class TranslateParams(OpParams):
    """Parameters for :func:`translate`.

    Attributes:
        new_initial_time: New ``initial_time`` value for the time axis.
    """

    kind: Literal["time_translate"] = "time_translate"
    new_initial_time: float

    chunkable_along: ClassVar[frozenset[str]] = frozenset({"time", "elements"})


def translate(ds: DataSource, p: TranslateParams) -> DataSource:
    return ds.with_time(ds.time.translate(p.new_initial_time))
