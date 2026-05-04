"""Filter specs for cfdmod's signal-processing pipeline.

Each spec is a Pydantic model in a discriminated union dispatched by
its ``kind`` literal. New filter types are added by:

1. Defining a new ``BaseModel`` subclass with a unique ``kind``
   ``Literal`` and the filter's params.
2. Adding the class to the :data:`FilterSpec` union.
3. Adding a dispatch branch in :func:`cfdmod.filters.core._apply_one`.
"""

from __future__ import annotations

__all__ = ["MovingAverageFilter", "FilterSpec"]

from typing import Annotated, Literal

from pydantic import BaseModel, Field


class MovingAverageFilter(BaseModel):
    """Centred moving-average smoothing of width ``window``.

    ``window`` is in the same units as the caller's ``dt`` (typically
    the input file's ``/meta/time_normalized`` values when invoked via
    :func:`cfdmod.filters.apply_filters_h5`). Internally the window is
    rounded to the nearest odd integer number of samples so the output
    stays aligned with the input timestamps; edges are handled by
    reflecting the signal so the output length matches the input.
    """

    kind: Literal["moving_average"] = "moving_average"
    window: Annotated[
        float,
        Field(gt=0, description="Window width in input time units."),
    ]


FilterSpec = Annotated[MovingAverageFilter, Field(discriminator="kind")]
# Add new filter classes to the union when introducing them, e.g.:
#   FilterSpec = Annotated[
#       MovingAverageFilter | LowPassFilter | ...,
#       Field(discriminator="kind"),
#   ]
