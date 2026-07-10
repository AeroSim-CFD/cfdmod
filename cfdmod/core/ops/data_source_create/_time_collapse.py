"""Shared helper for time-aggregating ops.

Ops that collapse the time axis (``statistics``, ``extreme_value``, ...)
all emit a :class:`DataSource` with the same element axis but
``n_timesteps == 0`` and a fresh field set. This helper centralises that
construction so the collapse contract (zeroed :class:`TimeAxis`, in-memory
field store, replaced metadata) lives in one place.
"""

from __future__ import annotations

__all__ = ["collapse_time_axis"]

from collections.abc import Mapping

import numpy as np

from cfdmod.adapters.memory import MemoryFieldStore
from cfdmod.core.data_source import DataSource
from cfdmod.core.field_meta import FieldMeta
from cfdmod.core.time_axis import TimeAxis


def collapse_time_axis(
    ds: DataSource,
    fields: Mapping[str, np.ndarray],
    field_meta: Mapping[str, FieldMeta],
) -> DataSource:
    """Return a copy of ``ds`` with its time axis collapsed.

    The result keeps ``ds``'s ``initial_time`` but has
    ``timestep_size == 0`` and ``n_timesteps == 0`` (time-aggregated),
    and carries ``fields`` / ``field_meta`` as its new field set.
    """
    new_time = TimeAxis(
        initial_time=ds.time.initial_time,
        timestep_size=0.0,
        n_timesteps=0,
    )
    return ds.model_copy(
        update={
            "time": new_time,
            "fields": MemoryFieldStore(dict(fields)),
            "field_meta": dict(field_meta),
        }
    )
