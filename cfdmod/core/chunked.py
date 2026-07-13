"""Chunked execution over the time axis, to bound peak memory.

The v3 ops advertise ``chunkable_along`` (see :class:`cfdmod.core.ops.OpParams`),
and almost the whole Cp -> Cf/Cm chain declares ``"time"``. This module is the
executor that actually honours it: it streams a time-preserving pipeline over
contiguous time windows and concatenates the per-window outputs, so peak memory
is ``O(n_elements * chunk)`` instead of ``O(n_elements * n_timesteps)``.

The win is real only when the pipeline *reduces* the element axis before the
concatenation (e.g. summing a per-triangle force to a per-floor coefficient):
the large intermediate arrays live only for one window at a time, while the
concatenated result is small. A pipeline that keeps the full element axis (e.g.
a bare ``force_contribution``) still bounds its transient allocations, but its
final output is the same size as the unchunked one.

Contract for a chunkable pipeline: it must be *time-length preserving* (input
window of ``k`` timesteps -> output of ``k`` timesteps) and its output element
axis / topology must be identical across windows. Every op it is built from must
declare ``"time"`` in ``chunkable_along`` -- use :func:`assert_time_chunkable`
to check that from a list of op params before running.
"""

from __future__ import annotations

__all__ = [
    "time_windows",
    "slice_time",
    "concat_time",
    "chunk_map_time",
    "assert_time_chunkable",
]

from typing import Callable, Iterable, Iterator, Sequence

import numpy as np

from cfdmod.adapters.memory import MemoryFieldStore
from cfdmod.core.data_source import DataSource
from cfdmod.core.time_axis import TimeAxis


def time_windows(n_timesteps: int, chunk_size: int) -> Iterator[slice]:
    """Yield contiguous ``slice`` objects covering ``range(n_timesteps)``.

    The last window is short when ``n_timesteps`` is not a multiple of
    ``chunk_size``. Raises for a non-positive ``chunk_size``.
    """
    if chunk_size <= 0:
        raise ValueError(f"chunk_size must be positive; got {chunk_size}")
    for start in range(0, max(n_timesteps, 0), chunk_size):
        yield slice(start, min(start + chunk_size, n_timesteps))


def slice_time(ds: DataSource, sl: slice) -> DataSource:
    """Return a windowed copy of ``ds`` over the time slice ``sl``.

    Fields are read for the window and materialised into a
    :class:`MemoryFieldStore`; topology / elements / groupings are preserved by
    reference. The window's :class:`TimeAxis` keeps the parent's
    ``timestep_size`` and normalization offset so normalized times stay
    continuous across windows.
    """
    if ds.time.is_time_aggregated:
        raise ValueError("cannot time-slice a time-aggregated data source")

    n_t = ds.time.n_timesteps
    start, stop, step = sl.indices(n_t)
    if step != 1:
        raise ValueError(f"slice_time expects a contiguous slice (step 1); got step {step}")
    n_win = max(stop - start, 0)
    if n_win == 0:
        raise ValueError(f"empty time window {sl!r} for n_timesteps={n_t}")

    fields = {name: np.asarray(ds.fields.read(name, time_slice=sl)) for name in ds.fields.keys()}
    win_time = TimeAxis(
        initial_time=float(ds.time.initial_time + ds.time.timestep_size * start),
        timestep_size=ds.time.timestep_size,
        n_timesteps=n_win,
        time_normalized_offset=ds.time.normalization_offset,
    )
    return ds._copy_validated(fields=MemoryFieldStore(fields), time=win_time)


def concat_time(parts: Sequence[DataSource]) -> DataSource:
    """Concatenate time-resolved outputs of successive windows along time.

    ``parts`` must share kind, element axis, field set and topology (only the
    time axis differs). 2-D fields are concatenated along axis 1; any 1-D
    (time-invariant) field is taken from the first part. The result carries the
    reconstructed full :class:`TimeAxis`.
    """
    if not parts:
        raise ValueError("concat_time needs at least one part")
    if len(parts) == 1:
        return parts[0]

    template = parts[0]
    names = list(template.field_names)
    total_t = int(sum(p.time.n_timesteps for p in parts))

    out_fields: dict[str, np.ndarray] = {}
    for name in names:
        arrs = [np.asarray(p.fields.read(name)) for p in parts]
        if arrs[0].ndim == 2:
            out_fields[name] = np.concatenate(arrs, axis=1)
        else:
            out_fields[name] = arrs[0]

    full_time = TimeAxis(
        initial_time=template.time.initial_time,
        timestep_size=template.time.timestep_size,
        n_timesteps=total_t,
        time_normalized_offset=template.time.normalization_offset,
    )
    return template._copy_validated(
        fields=MemoryFieldStore(out_fields),
        time=full_time,
        field_meta=dict(template.field_meta),
    )


def chunk_map_time(
    ds: DataSource,
    pipeline: Callable[[DataSource], DataSource],
    *,
    chunk_size: int | None,
) -> DataSource:
    """Run ``pipeline`` over time windows of ``ds`` and concatenate the results.

    ``pipeline`` is any single-arg ``DataSource -> DataSource`` callable that is
    time-length preserving and time-chunkable (see the module docstring). With
    ``chunk_size`` ``None`` or ``>= n_timesteps`` the pipeline runs once on the
    whole series (identical result, no chunking overhead), so this is a safe
    drop-in.
    """
    n_t = ds.time.n_timesteps
    if chunk_size is None or n_t == 0 or chunk_size >= n_t:
        return pipeline(ds)
    parts = [pipeline(slice_time(ds, sl)) for sl in time_windows(n_t, chunk_size)]
    return concat_time(parts)


def assert_time_chunkable(op_params: Iterable[object]) -> None:
    """Raise if any op in ``op_params`` does not declare ``"time"`` chunkability.

    Lets a caller validate that a pipeline built from a list of
    :class:`~cfdmod.core.ops.OpParams` is safe to run under
    :func:`chunk_map_time` before paying for any I/O.
    """
    offenders = [
        getattr(p, "kind", type(p).__name__)
        for p in op_params
        if "time" not in getattr(p, "chunkable_along", frozenset())
    ]
    if offenders:
        raise ValueError(
            "pipeline is not time-chunkable; these ops do not declare "
            f"time chunkability: {offenders}"
        )
