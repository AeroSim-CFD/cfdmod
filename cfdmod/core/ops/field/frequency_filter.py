"""Butterworth frequency-filter field op -- low / high / band pass & stop.

A zero-phase (default) or causal Butterworth filter applied along the
time axis. Like :mod:`cfdmod.core.ops.field.moving_average` it preserves
the element axis and the time axis and rewrites a single named field.

The sampling rate is derived from the data source: ``fs = 1 / dt`` where
``dt`` is ``ds.time.timestep_size``. Cutoff frequencies are given in Hz
and must lie strictly inside ``(0, fs/2)`` (the Nyquist frequency).

Zero-phase filtering (``zero_phase=True``, the default) uses
``scipy.signal.sosfiltfilt``: no phase lag, but the filter is non-causal
and its effective order is doubled. Set ``zero_phase=False`` for a
single-pass causal ``sosfilt`` (introduces the usual group delay).

scipy is already a runtime dependency (see ``inflow.py`` /
``hfpi/analysis.py``); the ``scipy.signal`` import is kept local to the
op function, mirroring how the RK45 solver isolates ``scipy.integrate``.
"""

from __future__ import annotations

__all__ = ["FrequencyFilterParams", "frequency_filter"]

from typing import Annotated, ClassVar, Literal

import numpy as np
from pydantic import Field, model_validator

from cfdmod.core.data_source import DataSource
from cfdmod.core.ops import OpParams

BType = Literal["lowpass", "highpass", "bandpass", "bandstop"]
_BAND_TYPES = frozenset({"bandpass", "bandstop"})


class FrequencyFilterParams(OpParams):
    """Parameters for :func:`frequency_filter`.

    Attributes:
        btype: Filter response type.
        cutoff: Cutoff frequency in Hz. A scalar for ``lowpass`` /
            ``highpass``; an ordered ``(low, high)`` pair for
            ``bandpass`` / ``bandstop``.
        order: Butterworth filter order. Defaults to 4.
        zero_phase: Zero-phase (forward-backward) filtering when True
            (default); single-pass causal filtering when False.
        field: Name of the field to filter. Defaults to ``"pressure"``.
        out: Optional output field name. Defaults to overwriting
            ``field`` in place.
    """

    kind: Literal["frequency_filter"] = "frequency_filter"
    btype: BType
    cutoff: float | tuple[float, float]
    order: Annotated[int, Field(gt=0, description="Butterworth filter order")] = 4
    zero_phase: bool = True
    field: str = "pressure"
    out: str | None = None

    chunkable_along: ClassVar[frozenset[str]] = frozenset({"elements"})

    @model_validator(mode="after")
    def _check_cutoff_arity(self) -> "FrequencyFilterParams":
        is_band = self.btype in _BAND_TYPES
        is_pair = isinstance(self.cutoff, tuple)
        if is_band and not is_pair:
            raise ValueError(
                f"{self.btype} requires a (low, high) cutoff pair, got {self.cutoff!r}"
            )
        if not is_band and is_pair:
            raise ValueError(
                f"{self.btype} requires a scalar cutoff frequency, got {self.cutoff!r}"
            )
        if is_pair:
            low, high = self.cutoff
            if not (low < high):
                raise ValueError(f"band cutoff must be ordered low < high, got {self.cutoff!r}")
            if low <= 0:
                raise ValueError(f"cutoff frequencies must be > 0, got {self.cutoff!r}")
        elif self.cutoff <= 0:
            raise ValueError(f"cutoff frequency must be > 0, got {self.cutoff!r}")
        return self


def frequency_filter(ds: DataSource, p: FrequencyFilterParams) -> DataSource:
    from scipy.signal import butter, sosfilt, sosfiltfilt

    if ds.time.is_time_aggregated:
        raise ValueError("frequency_filter requires a time-resolved data source")
    if ds.time.n_timesteps < 2:
        raise ValueError(
            "frequency_filter requires at least 2 timesteps to derive dt "
            f"(got n_timesteps={ds.time.n_timesteps})"
        )

    dt = float(ds.time.timestep_size)
    fs = 1.0 / dt
    nyquist = fs / 2.0

    cutoff = list(p.cutoff) if isinstance(p.cutoff, tuple) else [p.cutoff]
    for f in cutoff:
        if f >= nyquist:
            raise ValueError(
                f"cutoff {f} Hz must be below the Nyquist frequency {nyquist} Hz "
                f"(fs={fs} Hz from dt={dt})"
            )

    sos = butter(p.order, p.cutoff, btype=p.btype, fs=fs, output="sos")

    arr = np.asarray(ds.fields.read(p.field), dtype=np.float64)
    if arr.ndim != 2:
        raise ValueError(
            f"field {p.field!r} must be 2-D (n_elements, n_timesteps); got shape {arr.shape}"
        )

    if p.zero_phase:
        out = sosfiltfilt(sos, arr, axis=1)
    else:
        out = sosfilt(sos, arr, axis=1)
    out = np.ascontiguousarray(out, dtype=np.float64)

    target = p.out or p.field
    src_meta = ds.field_meta.get(p.field)
    if src_meta is not None and target != p.field:
        out_meta = src_meta.model_copy(update={"name": target})
    else:
        out_meta = src_meta
    return ds.with_field(target, out, meta=out_meta)
