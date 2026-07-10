"""Extreme-value / peak statistics op -- collapse the time axis.

Like :func:`cfdmod.core.ops.data_source_create.statistics.compute_statistics`
this op collapses the time axis (``n_timesteps == 0`` on the output) and
emits a single field of shape ``(n_elements,)``. It offers two peak
estimators that ``compute_statistics`` deliberately does not:

- ``method="gumbel"``: Gumbel extreme-value estimate. Each element's
  series is smoothed with a peak-duration moving window, split into
  ``n_subdivisions`` blocks, the per-block extremes are fit to a Gumbel
  distribution (``gumbel_r`` for ``max``, ``gumbel_l`` for ``min``), the
  fit is rescaled from the sub-window duration to the target
  ``event_duration``, and the ``non_exceedance_probability`` quantile is
  returned.
- ``method="peak_factor"``: ``mean +/- peak_factor * rms`` where ``rms``
  is the population standard deviation of the fluctuation (``ddof=0``,
  matching the legacy HFPI convention).

The pure-numpy helpers (:func:`moving_filter`,
:func:`reescale_event_duration_peak`, :func:`gumbel_extreme_value_1d`)
are ported from the orphaned ``cfdmod.hfpi.common`` and are importable
directly by recipes and tests. ``scipy.stats`` is imported lazily inside
the Gumbel fit, keeping the module's import cost scipy-free.
"""

from __future__ import annotations

__all__ = [
    "ExtremeValueParams",
    "extreme_value",
    "moving_filter",
    "reescale_event_duration_peak",
    "gumbel_extreme_value_1d",
]

from typing import Annotated, ClassVar, Literal

import numpy as np
from pydantic import Field, model_validator

from cfdmod.adapters.memory import MemoryFieldStore
from cfdmod.core.data_source import DataSource
from cfdmod.core.field_meta import FieldMeta
from cfdmod.core.ops import OpParams
from cfdmod.core.time_axis import TimeAxis


def moving_filter(hist_series: np.ndarray, dt: float, peak_duration: float) -> np.ndarray:
    """Peak-window moving average via valid-mode convolution.

    Unlike :func:`cfdmod.core.ops.field.moving_average.moving_average`
    (centred, edge-padded, same length), this shortens the series to
    ``len - window + 1`` and is *not* centred. The behaviour is intrinsic
    to the Gumbel subdivision scheme and is preserved byte-for-byte from
    the legacy HFPI implementation.
    """
    from scipy.signal import convolve

    window_size = max(int(peak_duration / dt), 1)
    kernel = np.ones(window_size) / window_size
    return convolve(hist_series, kernel, mode="valid")


def reescale_event_duration_peak(
    loc: float,
    scale: float,
    original_time: float,
    new_time: float,
    extreme_type: Literal["min", "max"],
) -> tuple[float, float]:
    """Rescale a Gumbel ``(loc, scale)`` from one event duration to another."""
    sign = 1 if extreme_type == "max" else -1
    new_scale = scale
    new_loc = loc + sign * scale * np.log(new_time / original_time)
    return new_loc, new_scale


def gumbel_extreme_value_1d(
    hist_series: np.ndarray,
    dt: float,
    peak_duration: float,
    event_duration: float,
    extreme_type: Literal["min", "max"],
    n_subdivisions: int = 10,
    non_exceedance_probability: float = 0.78,
) -> float:
    """Gumbel extreme-value estimate for a single 1-D series.

    Smooth -> subdivide -> fit Gumbel -> rescale to ``event_duration`` ->
    return the ``non_exceedance_probability`` quantile. Ported from
    ``cfdmod.hfpi.common.gumbel_extreme_value``.
    """
    if hist_series.ndim != 1:
        raise ValueError("gumbel_extreme_value_1d works only on 1-D arrays")

    smoothed_parent = moving_filter(hist_series, dt, peak_duration)
    sub_arrays = np.array_split(smoothed_parent, n_subdivisions)
    orig_time_duration = len(hist_series) * dt / n_subdivisions

    if extreme_type == "max":
        from scipy.stats import gumbel_r

        v_peak = np.array([np.max(sub_arr) for sub_arr in sub_arrays])
        loc, scale = gumbel_r.fit(v_peak)
        loc, scale = reescale_event_duration_peak(
            loc, scale, orig_time_duration, event_duration, extreme_type
        )
        return float(gumbel_r.ppf(non_exceedance_probability, loc=loc, scale=scale))

    from scipy.stats import gumbel_l

    v_peak = np.array([np.min(sub_arr) for sub_arr in sub_arrays])
    loc, scale = gumbel_l.fit(v_peak)
    loc, scale = reescale_event_duration_peak(
        loc, scale, orig_time_duration, event_duration, extreme_type
    )
    return float(gumbel_l.ppf(1 - non_exceedance_probability, loc=loc, scale=scale))


class ExtremeValueParams(OpParams):
    """Parameters for :func:`extreme_value`.

    Attributes:
        method: ``"gumbel"`` for the Gumbel extreme-value estimate,
            ``"peak_factor"`` for ``mean +/- peak_factor * rms``.
        extreme_type: ``"max"`` or ``"min"``.
        field: Source field to aggregate over time. Defaults to
            ``"pressure"``.
        out: Output field name. Defaults to ``"{method}_{extreme_type}"``.
        peak_duration: (gumbel) Smoothing window duration, in input time
            units.
        event_duration: (gumbel) Target event duration to rescale the fit
            to, in input time units.
        n_subdivisions: (gumbel) Number of blocks the smoothed series is
            split into for the block-maxima fit.
        non_exceedance_probability: (gumbel) Quantile of the fitted
            distribution to return.
        peak_factor: (peak_factor) Multiplier ``k`` on the fluctuation
            rms.
    """

    kind: Literal["extreme_value"] = "extreme_value"
    method: Literal["gumbel", "peak_factor"]
    extreme_type: Literal["min", "max"]
    field: str = "pressure"
    out: str | None = None

    # gumbel-only
    peak_duration: float | None = None
    event_duration: float | None = None
    n_subdivisions: Annotated[int, Field(gt=0)] = 10
    non_exceedance_probability: Annotated[float, Field(gt=0, lt=1)] = 0.78

    # peak_factor-only
    peak_factor: float | None = None

    chunkable_along: ClassVar[frozenset[str]] = frozenset({"elements"})
    replaces_fields: ClassVar[bool] = True

    @model_validator(mode="after")
    def _check_method_params(self) -> "ExtremeValueParams":
        if self.method == "gumbel":
            if self.peak_duration is None or self.event_duration is None:
                raise ValueError("method='gumbel' requires peak_duration and event_duration")
            if self.peak_factor is not None:
                raise ValueError("peak_factor is not valid for method='gumbel'")
        else:  # peak_factor
            if self.peak_factor is None:
                raise ValueError("method='peak_factor' requires peak_factor")
            extras = [
                name
                for name in ("peak_duration", "event_duration")
                if getattr(self, name) is not None
            ]
            if extras:
                raise ValueError(f"{extras} are not valid for method='peak_factor'")
        return self

    def _out_name(self) -> str:
        return self.out or f"{self.method}_{self.extreme_type}"

    def produced_fields(self) -> frozenset[str]:
        return frozenset({self._out_name()})


def extreme_value(ds: DataSource, p: ExtremeValueParams) -> DataSource:
    if ds.time.is_time_aggregated:
        raise ValueError("extreme_value requires a time-resolved data source")
    if ds.time.n_timesteps < 2:
        raise ValueError(
            f"extreme_value requires at least 2 timesteps (got n_timesteps={ds.time.n_timesteps})"
        )

    arr = np.asarray(ds.fields.read(p.field), dtype=np.float64)
    if arr.ndim != 2:
        raise ValueError(
            f"field {p.field!r} must be 2-D (n_elements, n_timesteps); got shape {arr.shape}"
        )

    if p.method == "peak_factor":
        mean = arr.mean(axis=1)
        rms = (arr - mean[:, None]).std(axis=1)
        sign = 1.0 if p.extreme_type == "max" else -1.0
        result = mean + sign * p.peak_factor * rms
    else:  # gumbel
        dt = float(ds.time.timestep_size)
        result = np.array(
            [
                gumbel_extreme_value_1d(
                    row,
                    dt=dt,
                    peak_duration=p.peak_duration,
                    event_duration=p.event_duration,
                    extreme_type=p.extreme_type,
                    n_subdivisions=p.n_subdivisions,
                    non_exceedance_probability=p.non_exceedance_probability,
                )
                for row in arr
            ]
        )

    out_name = p._out_name()
    src_meta = ds.field_meta.get(p.field)
    out_meta = (
        FieldMeta(name=out_name, unit=src_meta.unit, scale=src_meta.scale)
        if src_meta is not None
        else FieldMeta(name=out_name)
    )
    new_time = TimeAxis(
        initial_time=ds.time.initial_time,
        timestep_size=0.0,
        n_timesteps=0,
    )
    return ds.model_copy(
        update={
            "time": new_time,
            "fields": MemoryFieldStore({out_name: result}),
            "field_meta": {out_name: out_meta},
        }
    )
