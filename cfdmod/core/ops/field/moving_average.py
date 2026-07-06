"""Centred moving-average field op.

Reproduces ``cfdmod.pressure.filters.MovingAverageFilter`` exactly.

The window is given in input time units; it is rounded to the nearest
odd integer number of samples (so the output stays aligned with the
input timestamps), edges are handled with ``np.pad(mode="edge")`` so
the output length matches the input. This matches the legacy
implementation byte-for-byte.

The op operates on a single named field; chain multiple ops to filter
several fields. The data source's :class:`TimeAxis` is unchanged.
"""

from __future__ import annotations

__all__ = ["MovingAverageParams", "moving_average", "window_in_samples"]

from typing import Annotated, ClassVar, Literal

import numpy as np
from pydantic import Field

from cfdmod.core.data_source import DataSource
from cfdmod.core.ops import OpParams


class MovingAverageParams(OpParams):
    """Parameters for :func:`moving_average`.

    Attributes:
        window: Window width in the same time units as ``ds.time``.
            Rounded internally to the nearest odd integer number of
            samples.
        field: Name of the field to filter. Defaults to ``"pressure"``.
        out: Optional output field name. Defaults to overwriting
            ``field`` in place.
    """

    kind: Literal["moving_average"] = "moving_average"
    window: Annotated[float, Field(gt=0, description="Window width in input time units")]
    field: str = "pressure"
    out: str | None = None

    chunkable_along: ClassVar[frozenset[str]] = frozenset({"elements"})


def window_in_samples(window: float, dt: float) -> int:
    """Convert a time-units window to an odd integer sample count >= 1.

    Public for the pressure-filters delegation in ``pressure/filters.py``;
    the rounding behaviour must match byte-for-byte.
    """
    n = int(round(window / dt))
    if n < 1:
        n = 1
    if n % 2 == 0:
        n += 1
    return n


def _convolve_rows(data: np.ndarray, n: int) -> np.ndarray:
    """Per-row 1-D convolution with a uniform kernel.

    ``data`` shape is ``(n_elements, n_timesteps)``. Returns the same
    shape with reflect-edge padding so the output length matches the
    input.
    """
    if n == 1:
        return data
    kernel = np.ones(n, dtype=np.float64) / n
    pad = n // 2
    padded = np.pad(data, ((0, 0), (pad, pad)), mode="edge")
    out = np.empty_like(data, dtype=np.float64)
    for i in range(data.shape[0]):
        out[i, :] = np.convolve(padded[i, :], kernel, mode="valid")
    return out


def moving_average(ds: DataSource, p: MovingAverageParams) -> DataSource:
    if ds.time.is_time_aggregated:
        raise ValueError("moving_average requires a time-resolved data source")
    if ds.time.n_timesteps < 2:
        raise ValueError(
            "moving_average requires at least 2 timesteps to derive dt "
            f"(got n_timesteps={ds.time.n_timesteps})"
        )

    dt = float(ds.time.timestep_size)
    n = window_in_samples(p.window, dt)

    arr = np.asarray(ds.fields.read(p.field), dtype=np.float64)
    if arr.ndim != 2:
        raise ValueError(
            f"field {p.field!r} must be 2-D (n_elements, n_timesteps); " f"got shape {arr.shape}"
        )
    out = _convolve_rows(arr, n)

    target = p.out or p.field
    return ds.with_field(target, out, meta=ds.field_meta.get(p.field))
