"""Signal-processing filters for cfdmod timeseries files.

A filter is a 1-D signal-processing operation applied independently to
each triangle's time series. Filters are first-class pipeline steps:

    Cp -> [filter1, filter2, ...] -> filtered timeseries -> stats
or
    Cp -> filtered timeseries -> Cf / Cm (operate on the filtered Cp)

The user composes them via ``apply_filters(input_h5, output_h5,
filters=[...])``, which streams the input timeseries, applies the
chain in order, and writes a new timeseries H5 with the same on-disk
shape (``/Triangles + /Geometry``, ``/{group}/t{T}`` per timestep,
``/meta/...``). The applied filter chain is recorded in
``/processing_metadata`` so a downstream consumer can inspect what
was done.

Filter ``window`` values are in the same time units as the input
file's time axis (``/meta/time_normalized`` in cfdmod's layout). With
``CpConfig.normalize_time=False`` (the default) that is raw solver
time; with ``True`` it is convective time. The filter performs no
implicit unit conversion.

Adding a new filter type
------------------------
1. Add a new Pydantic model with a unique ``kind`` Literal and the
   filter's params.
2. Add it to the ``FilterSpec`` discriminated union below.
3. Add a branch in ``_apply_one`` that dispatches on ``kind``.
"""

from __future__ import annotations

__all__ = [
    "MovingAverageFilter",
    "FilterSpec",
    "apply_filters",
]

import pathlib
from typing import Annotated, Literal

import h5py
import numpy as np
from pydantic import BaseModel, Field

from cfdmod.io.xdmf import (
    get_pressure_keys,
    read_timeseries_meta,
    write_processing_metadata,
    write_temporal_xdmf,
    write_timeseries_geometry,
    write_timeseries_meta,
    write_timeseries_step,
)
from cfdmod.logger import logger

# ---------------------------------------------------------------------------
# Filter specs (flat discriminated union, dispatched by `kind`)
# ---------------------------------------------------------------------------


class MovingAverageFilter(BaseModel):
    """Centred moving-average smoothing of width ``window``.

    ``window`` is in the same units as the input file's time axis.
    Internally the window is rounded to the nearest odd integer number
    of samples (so the output stays aligned with the input timestamps);
    edges are handled by reflecting the signal so the output length
    matches the input.
    """

    kind: Literal["moving_average"] = "moving_average"
    window: Annotated[float, Field(gt=0, description="Window width in input time units")]


FilterSpec = Annotated[MovingAverageFilter, Field(discriminator="kind")]
# Add new filter classes to the union when introducing them, e.g.:
#   FilterSpec = Annotated[
#       MovingAverageFilter | LowPassFilter | ...,
#       Field(discriminator="kind"),
#   ]


# ---------------------------------------------------------------------------
# Application
# ---------------------------------------------------------------------------


def _window_in_samples(window: float, dt: float) -> int:
    """Convert a time-units window to an odd integer sample count >= 1."""
    n = int(round(window / dt))
    if n < 1:
        n = 1
    if n % 2 == 0:
        n += 1
    return n


def _apply_one(
    spec: MovingAverageFilter,
    data: np.ndarray,
    dt: float,
) -> np.ndarray:
    """Apply a single filter to a (n_time, n_tri) array along axis 0.

    Reflect-pad to keep the output length equal to the input length,
    so per-timestep alignment with /meta/time_steps is preserved.
    """
    if isinstance(spec, MovingAverageFilter):
        n = _window_in_samples(spec.window, dt)
        if n == 1:
            return data
        kernel = np.ones(n, dtype=np.float64) / n
        pad = n // 2
        # Vectorised per-column convolution via reflect padding + valid mode.
        padded = np.pad(data, ((pad, pad), (0, 0)), mode="edge")
        out = np.empty_like(data)
        for j in range(data.shape[1]):
            out[:, j] = np.convolve(padded[:, j], kernel, mode="valid")
        return out
    raise TypeError(f"unknown filter kind: {type(spec).__name__}")


def apply_filters(
    input_h5: pathlib.Path,
    output_h5: pathlib.Path,
    *,
    filters: list[FilterSpec],
    group: str = "cp",
) -> None:
    """Apply a chain of filters to one group of a timeseries H5 and write
    the result to ``output_h5`` with the same on-disk shape.

    Args:
        input_h5: Source timeseries H5 (e.g. ``cp.default.time_series.h5``).
        output_h5: Destination timeseries H5. Overwritten if it exists.
        filters: Sequence of filter specs applied in order.
        group: Coefficient group name inside the H5 (e.g. ``"cp"``,
            ``"cf_x"``, ``"cm_y"``).

    Output layout
    -------------
    - ``/Triangles + /Geometry``: copied verbatim from ``input_h5``.
    - ``/meta/time_steps + /meta/time_normalized``: copied verbatim
      from ``input_h5`` (filter preserves the time axis; only sample
      values change).
    - ``/{group}/t{T}``: the filtered series, one dataset per timestep.
    - ``/processing_metadata``: records the applied filter chain plus
      the input file path (round-trip via
      :func:`cfdmod.io.read_processing_metadata`).
    - A temporal XDMF sibling is written next to ``output_h5`` so
      ParaView can animate the filtered field.
    """
    if not filters:
        raise ValueError("apply_filters: filters list is empty (no-op)")
    output_h5 = pathlib.Path(output_h5)
    if output_h5.exists():
        output_h5.unlink()

    keys = get_pressure_keys(input_h5, group)
    if not keys:
        raise ValueError(f"{input_h5}:/{group} has no timesteps")

    # Load full series into memory: needed because moving-average (and any
    # future temporal filter) requires the whole time axis per triangle.
    # For the worst-case 150k tri x 10k step Cp this is ~12 GB at f64;
    # downstream chunking can be added if it bites in practice.
    with h5py.File(input_h5, "r") as f:
        triangles = f["Triangles"][:]
        vertices = f["Geometry"][:]
        data = np.stack([f[group][k][:].astype(np.float64) for _, k in keys])

    meta = read_timeseries_meta(input_h5)
    time_steps = np.asarray(meta["time_steps"], dtype=np.float64)
    time_normalized = np.asarray(meta["time_normalized"], dtype=np.float64)

    # Filter window is in input time-axis units, which matches /meta/time_normalized
    # by contract. Use that to get dt; fall back to time_steps if the file
    # was authored without time_normalized written separately.
    axis = time_normalized if time_normalized.size == data.shape[0] else time_steps
    if axis.size < 2:
        raise ValueError("filter requires at least 2 timesteps to derive dt")
    dts = np.diff(axis)
    dt = float(dts.mean())
    if not np.allclose(dts, dt, rtol=1e-3):
        raise ValueError(
            "filter requires uniform timestep spacing; "
            f"detected jitter (dt min={dts.min()}, max={dts.max()})"
        )

    logger.info(
        f"apply_filters: {len(filters)} filter(s) on {input_h5} -> {output_h5} "
        f"(n_time={data.shape[0]}, n_tri={data.shape[1]}, dt={dt:.6g})"
    )
    out = data
    for spec in filters:
        logger.info(f"  applying {type(spec).__name__}({spec.model_dump()})")
        out = _apply_one(spec, out, dt)

    # Write filtered output
    write_timeseries_geometry(output_h5, triangles, vertices)
    for i, (_, k) in enumerate(keys):
        write_timeseries_step(output_h5, group, k, out[i])
    write_timeseries_meta(output_h5, time_steps, time_normalized)
    write_temporal_xdmf(output_h5, output_h5.with_suffix(".xdmf"), group)

    write_processing_metadata(
        output_h5,
        "/",
        {"filters": [spec.model_dump() for spec in filters]},
        extra={"source_h5": str(input_h5), "group": group},
    )
    logger.info(f"apply_filters: wrote {output_h5}")
