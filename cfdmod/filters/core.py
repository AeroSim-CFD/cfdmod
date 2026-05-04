"""Numpy-based filter application core.

Pure-numpy entry point. No file I/O. Wraps each filter type's
implementation behind a flat :func:`apply_filters` that the H5
wrapper (and any external caller) delegates to.
"""

from __future__ import annotations

__all__ = ["apply_filters", "_window_in_samples"]

import numpy as np

from cfdmod.filters.specs import FilterSpec, MovingAverageFilter


def _window_in_samples(window: float, dt: float) -> int:
    """Convert a time-units window to an odd integer sample count >= 1."""
    n = int(round(window / dt))
    if n < 1:
        n = 1
    if n % 2 == 0:
        n += 1
    return n


def apply_filters(
    data: np.ndarray,
    dt: float,
    filters: list[FilterSpec],
) -> np.ndarray:
    """Apply a chain of filters along axis 0 of ``data``.

    Pure-numpy: no file I/O, no metadata. Use this when the timeseries
    is already in memory (notebook, custom pipeline, or any source
    that is not cfdmod's standard H5 layout). The H5-based wrapper
    :func:`cfdmod.filters.apply_filters_h5` calls this function under
    the hood.

    Args:
        data: ``(n_time,)`` 1D or ``(n_time, n_features)`` 2D array.
            Filters are applied along axis 0; columns are independent.
            Promoted to float64 internally; demoted back to the input
            dtype on return so caller-side arrays keep their precision.
        dt: Sample spacing in the same units as the filter ``window``.
            Caller is responsible for ensuring the time axis is
            uniformly spaced.
        filters: Sequence of filter specs applied in left-to-right
            order. Empty list raises ``ValueError``.

    Returns:
        Filtered array of the same shape and dtype as ``data``.

    Raises:
        ValueError: ``filters`` empty, ``dt`` non-positive, ``data``
            not 1D/2D, or fewer than 2 samples along axis 0.
    """
    if not filters:
        raise ValueError("apply_filters: filters list is empty (no-op)")
    if dt <= 0:
        raise ValueError(f"apply_filters: dt must be > 0; got {dt}")

    arr = np.asarray(data)
    if arr.ndim not in (1, 2):
        raise ValueError(
            f"apply_filters: data must be 1D or 2D; got {arr.ndim}D shape {arr.shape}"
        )
    if arr.shape[0] < 2:
        raise ValueError(
            f"apply_filters: need at least 2 timesteps along axis 0; got {arr.shape[0]}"
        )

    in_dtype = arr.dtype
    if arr.ndim == 1:
        work = arr.astype(np.float64).reshape(-1, 1)
        was_1d = True
    else:
        work = arr.astype(np.float64)
        was_1d = False

    for spec in filters:
        work = _apply_one(spec, work, dt)

    out = work[:, 0] if was_1d else work
    return out.astype(in_dtype, copy=False)


def _apply_one(spec, data: np.ndarray, dt: float) -> np.ndarray:
    if isinstance(spec, MovingAverageFilter):
        n = _window_in_samples(spec.window, dt)
        if n == 1:
            return data
        kernel = np.ones(n, dtype=np.float64) / n
        pad = n // 2
        padded = np.pad(data, ((pad, pad), (0, 0)), mode="edge")
        out = np.empty_like(data)
        for j in range(data.shape[1]):
            out[:, j] = np.convolve(padded[:, j], kernel, mode="valid")
        return out
    raise TypeError(f"unknown filter kind: {type(spec).__name__}")
