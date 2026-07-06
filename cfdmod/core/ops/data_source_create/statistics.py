"""Statistics op -- collapse the time axis of a data source.

Produces a new :class:`DataSource` with the same element axis but
``n_timesteps == 0`` (time-aggregated). Each requested statistic
becomes a separate field on the output, named after the statistic
itself (``"mean"``, ``"rms"``, ``"peak_max"``, ...).

Supported statistics
--------------------

- ``mean``: arithmetic mean over time.
- ``rms``: standard deviation of the fluctuation (sample, ddof=1). Named
  ``rms`` to match the wind-engineering ``Cp_rms`` convention; note this
  is ``std``, not ``sqrt(mean(x^2))``, and the two differ when the mean
  is nonzero.
- ``min`` / ``max``: raw sample minimum / maximum over time.
- ``peak_min`` / ``peak_max``: currently identical to ``min`` / ``max``
  (raw extremes), kept as separate names for parity with the odt
  vocabulary; they are *not* peak-factor / extreme-value estimates.
- ``skewness`` / ``kurtosis``: third / excess fourth moments per
  element.

For each requested stat, the output field is shape ``(n_elements,)``.
"""

from __future__ import annotations

__all__ = ["StatisticsParams", "compute_statistics", "STAT_KINDS"]

from typing import ClassVar, Literal

import numpy as np

from cfdmod.adapters.memory import MemoryFieldStore
from cfdmod.core.data_source import DataSource
from cfdmod.core.field_meta import FieldMeta
from cfdmod.core.ops import OpParams
from cfdmod.core.time_axis import TimeAxis

STAT_KINDS = Literal[
    "mean",
    "rms",
    "min",
    "max",
    "peak_min",
    "peak_max",
    "skewness",
    "kurtosis",
]


class StatisticsParams(OpParams):
    """Parameters for :func:`compute_statistics`.

    Attributes:
        kinds: Statistics to compute, one output field per kind.
        field: Source field to aggregate over time. Defaults to
            ``"pressure"``.
    """

    kind: Literal["statistics"] = "statistics"
    kinds: list[STAT_KINDS]
    field: str = "pressure"

    chunkable_along: ClassVar[frozenset[str]] = frozenset({"elements"})


def _stat(arr: np.ndarray, name: str) -> np.ndarray:
    if name == "mean":
        return arr.mean(axis=1)
    if name == "rms":
        return arr.std(axis=1, ddof=1) if arr.shape[1] > 1 else np.zeros(arr.shape[0])
    if name in ("min", "peak_min"):
        return arr.min(axis=1)
    if name in ("max", "peak_max"):
        return arr.max(axis=1)
    if name == "skewness":
        m = arr.mean(axis=1, keepdims=True)
        var = ((arr - m) ** 2).mean(axis=1)
        m3 = ((arr - m) ** 3).mean(axis=1)
        with np.errstate(divide="ignore", invalid="ignore"):
            return np.where(var > 0, m3 / var**1.5, 0.0)
    if name == "kurtosis":
        m = arr.mean(axis=1, keepdims=True)
        var = ((arr - m) ** 2).mean(axis=1)
        m4 = ((arr - m) ** 4).mean(axis=1)
        with np.errstate(divide="ignore", invalid="ignore"):
            return np.where(var > 0, m4 / var**2 - 3.0, 0.0)
    raise ValueError(f"unknown statistic kind {name!r}")


def compute_statistics(ds: DataSource, p: StatisticsParams) -> DataSource:
    if ds.time.is_time_aggregated:
        raise ValueError("compute_statistics requires a time-resolved data source")

    arr = np.asarray(ds.fields.read(p.field), dtype=np.float64)
    if arr.ndim != 2:
        raise ValueError(
            f"field {p.field!r} must be 2-D (n_elements, n_timesteps); " f"got shape {arr.shape}"
        )

    out_arrays: dict[str, np.ndarray] = {}
    out_meta: dict[str, FieldMeta] = {}
    src_meta = ds.field_meta.get(p.field)
    for kind in p.kinds:
        out_arrays[kind] = _stat(arr, kind)
        out_meta[kind] = (
            FieldMeta(name=kind, unit=src_meta.unit, scale=src_meta.scale)
            if src_meta is not None
            else FieldMeta(name=kind)
        )

    new_time = TimeAxis(
        initial_time=ds.time.initial_time,
        timestep_size=0.0,
        n_timesteps=0,
    )
    return ds.model_copy(
        update={
            "time": new_time,
            "fields": MemoryFieldStore(out_arrays),
            "field_meta": out_meta,
        }
    )
