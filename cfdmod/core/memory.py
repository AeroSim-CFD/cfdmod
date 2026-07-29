"""Turn a RAM budget into a time-chunk size.

:mod:`cfdmod.core.chunked` can stream a time-preserving pipeline over windows of
the time axis so peak memory is ``O(n_elements * chunk)`` rather than
``O(n_elements * n_timesteps)``. It needs a ``chunk_size`` in timesteps, and
that is the wrong question to put to a caller: nobody knows what 4096 timesteps
costs. "I have 8 GB" is answerable.

The arithmetic is deliberately simple, and deliberately an estimate:

    one time column   = n_elements * itemsize          bytes
    live at any time  = n_live_arrays columns          (input + intermediates)
    chunk_size        = budget / (n_elements * itemsize * n_live_arrays)

``n_live_arrays`` is the count of time-resolved arrays alive at the widest point
of the pipeline. It is not knowable exactly -- numpy temporaries inside an op,
copies an adapter makes on read -- so this is a *lower bound on the cost* and
therefore an upper bound on a safe chunk size. Leave headroom in the budget
rather than trusting the estimate to the byte; :data:`DEFAULT_SAFETY_FACTOR`
applies some by default.

Nothing here allocates or measures. It is arithmetic over shapes, so it is
cheap enough to call before deciding whether to run at all.
"""

from __future__ import annotations

__all__ = [
    "ChunkPlan",
    "DEFAULT_SAFETY_FACTOR",
    "bytes_per_timestep",
    "estimate_peak_bytes",
    "plan_chunking",
    "suggest_chunk_size",
]

from dataclasses import dataclass

import numpy as np

from cfdmod.core.dtypes import FIELD_DTYPE

DEFAULT_SAFETY_FACTOR = 0.8
"""Fraction of the stated budget the planner will actually spend.

The estimate counts whole arrays and cannot see numpy's temporaries, so
spending the budget exactly is how a run that was "planned to fit" does not.
"""


@dataclass(frozen=True)
class ChunkPlan:
    """A chunk size plus everything needed to explain it.

    Returning this rather than a bare int is the point: when a run picks
    ``chunk_size=1`` the immediate question is why, and a plain integer cannot
    answer it.

    Attributes:
        chunk_size: Timesteps per window. Equal to ``n_timesteps`` when the
            whole series fits, in which case chunking is a no-op.
        n_timesteps: The full time axis length this was planned against.
        n_elements: Elements on the source.
        n_live_arrays: How many time-resolved arrays the plan assumed are
            simultaneously live.
        itemsize: Bytes per value.
        estimated_peak_bytes: What one window is expected to cost.
        budget_bytes: The budget the plan was derived from, or ``None`` when
            the chunk size was given directly.
        clamped: True when the arithmetic asked for something outside
            ``[1, n_timesteps]`` and the plan was pinned to the bound. A
            ``clamped`` plan at ``chunk_size == 1`` means the budget does not
            actually fit a single timestep and the run will exceed it.
    """

    chunk_size: int
    n_timesteps: int
    n_elements: int
    n_live_arrays: int
    itemsize: int
    estimated_peak_bytes: int
    budget_bytes: int | None = None
    clamped: bool = False

    @property
    def is_chunked(self) -> bool:
        """Whether this plan actually splits the time axis."""
        return 0 < self.chunk_size < self.n_timesteps

    @property
    def exceeds_budget(self) -> bool:
        """True when even one timestep does not fit the budget."""
        return self.budget_bytes is not None and self.estimated_peak_bytes > self.budget_bytes

    def describe(self) -> str:
        """One line for a log or a CLI, in units a human reads."""
        peak_mb = self.estimated_peak_bytes / 1e6
        if not self.is_chunked:
            return (
                f"time chunking: off (whole series of {self.n_timesteps} steps in one pass, "
                f"est. peak {peak_mb:.0f} MB)"
            )
        n_windows = -(-self.n_timesteps // self.chunk_size)
        note = " -- BUDGET EXCEEDED, one timestep does not fit" if self.exceeds_budget else ""
        return (
            f"time chunking: {self.chunk_size} steps x {n_windows} windows "
            f"({self.n_elements} elements, {self.n_live_arrays} live arrays, "
            f"est. peak {peak_mb:.0f} MB){note}"
        )


def _itemsize(dtype) -> int:
    return int(np.dtype(dtype).itemsize)


def bytes_per_timestep(
    n_elements: int,
    *,
    n_live_arrays: int = 1,
    dtype=FIELD_DTYPE,
) -> int:
    """Bytes one timestep occupies across every simultaneously live array."""
    if n_elements < 0:
        raise ValueError(f"n_elements must be non-negative; got {n_elements}")
    if n_live_arrays < 1:
        raise ValueError(f"n_live_arrays must be at least 1; got {n_live_arrays}")
    return int(n_elements) * _itemsize(dtype) * int(n_live_arrays)


def estimate_peak_bytes(
    n_elements: int,
    chunk_size: int,
    *,
    n_live_arrays: int = 1,
    dtype=FIELD_DTYPE,
) -> int:
    """Estimated peak bytes held while processing one window of ``chunk_size`` steps."""
    if chunk_size < 0:
        raise ValueError(f"chunk_size must be non-negative; got {chunk_size}")
    return bytes_per_timestep(n_elements, n_live_arrays=n_live_arrays, dtype=dtype) * int(
        chunk_size
    )


def suggest_chunk_size(
    n_elements: int,
    n_timesteps: int,
    budget_bytes: int,
    *,
    n_live_arrays: int = 1,
    dtype=FIELD_DTYPE,
    safety_factor: float = DEFAULT_SAFETY_FACTOR,
) -> int:
    """Largest window that is expected to fit ``budget_bytes``.

    Clamped to ``[1, n_timesteps]``: never zero (a zero-step window processes
    nothing) and never more than the series has. See :func:`plan_chunking` when
    you need to know *whether* it was clamped.
    """
    return plan_chunking(
        n_elements,
        n_timesteps,
        budget_bytes=budget_bytes,
        n_live_arrays=n_live_arrays,
        dtype=dtype,
        safety_factor=safety_factor,
    ).chunk_size


def plan_chunking(
    n_elements: int,
    n_timesteps: int,
    *,
    budget_bytes: int | None = None,
    chunk_size: int | None = None,
    n_live_arrays: int = 1,
    dtype=FIELD_DTYPE,
    safety_factor: float = DEFAULT_SAFETY_FACTOR,
) -> ChunkPlan:
    """Build a :class:`ChunkPlan` from a budget, or price an explicit chunk size.

    Exactly one of ``budget_bytes`` / ``chunk_size`` may be given. With neither,
    the plan is the whole series in one pass, still carrying a peak estimate --
    which is useful on its own: it is the number that says whether chunking is
    worth turning on.
    """
    if budget_bytes is not None and chunk_size is not None:
        raise ValueError("pass budget_bytes or chunk_size, not both")
    if n_timesteps < 0:
        raise ValueError(f"n_timesteps must be non-negative; got {n_timesteps}")
    if not 0 < safety_factor <= 1:
        raise ValueError(f"safety_factor must be in (0, 1]; got {safety_factor}")

    itemsize = _itemsize(dtype)
    unit = bytes_per_timestep(n_elements, n_live_arrays=n_live_arrays, dtype=dtype)
    clamped = False

    if chunk_size is not None:
        if chunk_size < 1:
            raise ValueError(f"chunk_size must be positive; got {chunk_size}")
        resolved = min(int(chunk_size), n_timesteps) if n_timesteps else int(chunk_size)
        clamped = resolved != int(chunk_size)
    elif budget_bytes is not None:
        if budget_bytes <= 0:
            raise ValueError(f"budget_bytes must be positive; got {budget_bytes}")
        # unit == 0 only when there are no elements, in which case any window fits.
        raw = n_timesteps if unit == 0 else int(budget_bytes * safety_factor) // unit
        resolved = max(1, min(raw, n_timesteps)) if n_timesteps else max(1, raw)
        clamped = raw != resolved
    else:
        resolved = n_timesteps

    return ChunkPlan(
        chunk_size=resolved,
        n_timesteps=int(n_timesteps),
        n_elements=int(n_elements),
        n_live_arrays=int(n_live_arrays),
        itemsize=itemsize,
        estimated_peak_bytes=unit * resolved,
        budget_bytes=budget_bytes,
        clamped=clamped,
    )
