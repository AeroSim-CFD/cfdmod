"""Time-rescale op.

Multiplies ``initial_time`` and ``timestep_size`` by a positive factor
(useful for converting solver time to convective time, or seconds to
ms). Pure metadata update; field arrays are not read.
"""

from __future__ import annotations

__all__ = ["RescaleTimeParams", "rescale"]

from typing import ClassVar, Literal

from cfdmod.core.data_source import DataSource
from cfdmod.core.ops import OpParams


class RescaleTimeParams(OpParams):
    """Parameters for :func:`rescale`.

    Attributes:
        factor: Positive scalar multiplied into both ``initial_time``
            and ``timestep_size``.
    """

    kind: Literal["time_rescale"] = "time_rescale"
    factor: float

    chunkable_along: ClassVar[frozenset[str]] = frozenset({"time", "elements"})


def rescale(ds: DataSource, p: RescaleTimeParams) -> DataSource:
    return ds.with_time(ds.time.rescale(p.factor))
