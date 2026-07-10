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
were ported from the former legacy HFPI ``common`` module. The port is
faithful except for one deliberate correction: the Gumbel sub-window
duration is derived from the smoothed series length rather than the raw
one (see :func:`gumbel_extreme_value_1d`), so results differ from the
legacy by ~window/record. They are importable directly by recipes and tests.
``scipy.stats`` is imported lazily inside the Gumbel fit, keeping the
module's import cost scipy-free.
"""

from __future__ import annotations

__all__ = [
    "ExtremeValueParams",
    "extreme_value",
    "moving_filter",
    "reescale_event_duration_peak",
    "gumbel_extreme_value_1d",
]

from typing import ClassVar, Literal

import numpy as np
from pydantic import model_validator

from cfdmod.core.data_source import DataSource
from cfdmod.core.field_meta import FieldMeta
from cfdmod.core.ops import OpParams
from cfdmod.core.ops.data_source_create._time_collapse import collapse_time_axis


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
    if window_size > hist_series.size:
        raise ValueError(
            f"peak-window size {window_size} (peak_duration={peak_duration}, dt={dt}) "
            f"exceeds the series length {hist_series.size}; use a shorter peak_duration "
            "or a longer record"
        )
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
    if smoothed_parent.size < n_subdivisions:
        raise ValueError(
            f"smoothed series length {smoothed_parent.size} is shorter than "
            f"n_subdivisions={n_subdivisions} (series length {hist_series.size}, "
            f"peak_duration={peak_duration}, dt={dt}); use a longer record, a smaller "
            "peak_duration, or fewer subdivisions"
        )
    sub_arrays = np.array_split(smoothed_parent, n_subdivisions)
    # The block-maxima come from the smoothed series (valid-mode
    # convolution drops the w-1 partial-window edge samples), so the
    # sub-window duration is derived from its length, not the raw series.
    # This differs from cfdmod.hfpi.common, which used the raw length and
    # thereby overstated the reference duration -- biasing the
    # event-duration rescale by ~window/record.
    orig_time_duration = smoothed_parent.size * dt / n_subdivisions

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
            split into for the block-maxima fit. Defaults to 10.
        non_exceedance_probability: (gumbel) Quantile of the fitted
            distribution to return. Defaults to 0.78.
        peak_factor: (peak_factor) Multiplier ``k`` on the fluctuation
            rms.

    The method-specific parameters default to ``None`` so that setting one
    under the wrong ``method`` is a validation error rather than a silently
    ignored value. The gumbel defaults (10 subdivisions, 0.78 quantile) are
    applied in :func:`extreme_value`, not here.
    """

    kind: Literal["extreme_value"] = "extreme_value"
    method: Literal["gumbel", "peak_factor"]
    extreme_type: Literal["min", "max"]
    field: str = "pressure"
    out: str | None = None

    # gumbel-only
    peak_duration: float | None = None
    event_duration: float | None = None
    n_subdivisions: int | None = None
    non_exceedance_probability: float | None = None

    # peak_factor-only
    peak_factor: float | None = None

    chunkable_along: ClassVar[frozenset[str]] = frozenset({"elements"})
    replaces_fields: ClassVar[bool] = True

    _GUMBEL_ONLY: ClassVar[tuple[str, ...]] = (
        "peak_duration",
        "event_duration",
        "n_subdivisions",
        "non_exceedance_probability",
    )

    @model_validator(mode="after")
    def _check_method_params(self) -> "ExtremeValueParams":
        if self.method == "gumbel":
            if self.peak_duration is None or self.event_duration is None:
                raise ValueError("method='gumbel' requires peak_duration and event_duration")
            if self.peak_duration <= 0 or self.event_duration <= 0:
                raise ValueError(
                    f"peak_duration and event_duration must be > 0, got "
                    f"peak_duration={self.peak_duration}, event_duration={self.event_duration}"
                )
            if self.peak_factor is not None:
                raise ValueError("peak_factor is not valid for method='gumbel'")
            if self.n_subdivisions is not None and self.n_subdivisions <= 0:
                raise ValueError(f"n_subdivisions must be > 0, got {self.n_subdivisions}")
            if self.non_exceedance_probability is not None and not (
                0 < self.non_exceedance_probability < 1
            ):
                raise ValueError(
                    f"non_exceedance_probability must be in (0, 1), "
                    f"got {self.non_exceedance_probability}"
                )
        else:  # peak_factor
            if self.peak_factor is None:
                raise ValueError("method='peak_factor' requires peak_factor")
            extras = [name for name in self._GUMBEL_ONLY if getattr(self, name) is not None]
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
        # np.std centres the data itself, so std of the fluctuation is
        # just std of the raw series; matches statistics.py rms (ddof=0).
        sign = 1.0 if p.extreme_type == "max" else -1.0
        result = arr.mean(axis=1) + sign * p.peak_factor * arr.std(axis=1)
    else:  # gumbel
        dt = float(ds.time.timestep_size)
        n_subdivisions = p.n_subdivisions if p.n_subdivisions is not None else 10
        non_exceedance = (
            p.non_exceedance_probability if p.non_exceedance_probability is not None else 0.78
        )
        result = np.array(
            [
                gumbel_extreme_value_1d(
                    row,
                    dt=dt,
                    peak_duration=p.peak_duration,
                    event_duration=p.event_duration,
                    extreme_type=p.extreme_type,
                    n_subdivisions=n_subdivisions,
                    non_exceedance_probability=non_exceedance,
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
    return collapse_time_axis(ds, {out_name: result}, {out_name: out_meta})
