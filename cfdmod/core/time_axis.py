"""Affine :class:`TimeAxis` -- the time index for a :class:`DataSource`.

The proposal is explicit (issue #131): the time axis is reconstructable
from three numbers. Storing the full array is wrong because it grows
linearly with the simulation length and offers no information you do
not already have.

::

    target_time = initial_time + timestep_size * time_index

Time ops (``window_selection``, ``translate``, ``rescale``) mutate the
three numbers and never resample. Resampling is a *field* op and runs
through the field store.

The class is frozen. Functional updates produce a new instance via
:meth:`window`, :meth:`translate`, :meth:`rescale`.
"""

from __future__ import annotations

__all__ = ["TimeAxis"]

from typing import Annotated

import numpy as np
from pydantic import BaseModel, ConfigDict, Field, model_validator


class TimeAxis(BaseModel):
    """Affine time index.

    A time-aggregated :class:`DataSource` (statistics output, a single
    snapshot) uses ``n_timesteps == 0`` to mean "no time axis". The
    field store side mirrors this: shape ``(n_elements,)`` instead of
    ``(n_elements, n_timesteps)``.

    Attributes:
        initial_time: Time at index 0.
        timestep_size: Timestep delta. Must be > 0 when
            ``n_timesteps > 0``; otherwise the value is irrelevant
            (kept at 0.0 by convention).
        n_timesteps: Number of timesteps. ``0`` -> no time axis.
        time_normalized_offset: Optional offset applied when reporting
            "normalized" time (i.e. simulation time minus a chosen
            reference). Mirrors the existing ``meta/time_normalized``
            convention in cfdmod's h5 layout. Defaults to
            ``initial_time`` so that without explicit normalization,
            ``time_normalized[0] == 0``.
    """

    model_config = ConfigDict(frozen=True, arbitrary_types_allowed=True)

    initial_time: Annotated[float, Field(description="Simulation time at index 0.")]
    timestep_size: Annotated[
        float,
        Field(ge=0.0, description="Timestep delta; 0 allowed only when n_timesteps == 0."),
    ]
    n_timesteps: Annotated[int, Field(ge=0, description="Number of timesteps (0 = no time axis).")]
    time_normalized_offset: Annotated[
        float | None,
        Field(
            default=None,
            description=(
                "Offset for reporting normalized time. None -> use initial_time, so "
                "time_normalized[0] == 0."
            ),
        ),
    ] = None

    @model_validator(mode="after")
    def _check_step(self) -> "TimeAxis":
        if self.n_timesteps > 0 and self.timestep_size <= 0:
            raise ValueError(
                "timestep_size must be > 0 when n_timesteps > 0 "
                f"(got n_timesteps={self.n_timesteps}, timestep_size={self.timestep_size})"
            )
        return self

    @property
    def is_time_aggregated(self) -> bool:
        """True for stats outputs / single snapshots (no time axis)."""
        return self.n_timesteps == 0

    @property
    def normalization_offset(self) -> float:
        """Resolved offset used for normalized time."""
        return self.initial_time if self.time_normalized_offset is None else self.time_normalized_offset

    def times(self) -> np.ndarray:
        """Materialise the full time array. Avoid in hot paths."""
        if self.is_time_aggregated:
            return np.empty(0, dtype=np.float64)
        return self.initial_time + self.timestep_size * np.arange(self.n_timesteps, dtype=np.float64)

    def times_normalized(self) -> np.ndarray:
        """Materialise the full normalized time array."""
        return self.times() - self.normalization_offset

    def time_at(self, index: int) -> float:
        """Time corresponding to a single index. Negative index counts from the end."""
        if self.is_time_aggregated:
            raise ValueError("Time-aggregated axis has no time samples.")
        if index < 0:
            index += self.n_timesteps
        if not 0 <= index < self.n_timesteps:
            raise IndexError(f"Time index {index} out of range (n_timesteps={self.n_timesteps})")
        return self.initial_time + self.timestep_size * index

    def index_for_time(self, t: float) -> int:
        """Round-to-nearest index for ``t``. Useful for window selection."""
        if self.is_time_aggregated:
            raise ValueError("Time-aggregated axis has no time samples.")
        idx = int(round((t - self.initial_time) / self.timestep_size))
        return max(0, min(self.n_timesteps - 1, idx))

    # ----- Time ops: each returns a new TimeAxis ---------------------------------

    def window(self, start: float, end: float) -> tuple["TimeAxis", slice]:
        """Time-axis form of "window selection".

        Returns the new :class:`TimeAxis` *plus* the index slice
        callers can apply to their field arrays. The slice is open on
        the right, matching numpy convention.
        """
        if self.is_time_aggregated:
            raise ValueError("Cannot window a time-aggregated axis.")
        if end < start:
            raise ValueError(f"window end={end} < start={start}")
        i0 = self.index_for_time(start)
        i1 = self.index_for_time(end)
        if i1 < i0:
            i1 = i0
        index_slice = slice(i0, i1 + 1)
        new_axis = self.model_copy(
            update={
                "initial_time": self.time_at(i0),
                "n_timesteps": index_slice.stop - index_slice.start,
            }
        )
        return new_axis, index_slice

    def translate(self, new_initial_time: float) -> "TimeAxis":
        """Set ``initial_time``; keep step size and length."""
        return self.model_copy(update={"initial_time": float(new_initial_time)})

    def rescale(self, factor: float) -> "TimeAxis":
        """Multiply ``initial_time`` and ``timestep_size`` by ``factor``."""
        if factor <= 0:
            raise ValueError(f"rescale factor must be > 0 (got {factor})")
        return self.model_copy(
            update={
                "initial_time": self.initial_time * factor,
                "timestep_size": self.timestep_size * factor,
            }
        )

    def with_normalization_offset(self, offset: float) -> "TimeAxis":
        """Override the normalization offset (defaults to ``initial_time``)."""
        return self.model_copy(update={"time_normalized_offset": float(offset)})
