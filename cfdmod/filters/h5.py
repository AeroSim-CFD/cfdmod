"""H5-backed wrapper for filter application.

File-in / file-out flow over cfdmod's standard timeseries H5 layout.
Loads the source group, calls the numpy core
(:func:`cfdmod.filters.apply_filters`), writes the result with the
same on-disk shape, and records the applied chain under
``/processing_metadata`` so the lineage is self-describing.
"""

from __future__ import annotations

__all__ = ["apply_filters_h5"]

import pathlib

import h5py
import numpy as np

from cfdmod.filters.core import apply_filters
from cfdmod.filters.specs import FilterSpec
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


def apply_filters_h5(
    input_h5: pathlib.Path,
    output_h5: pathlib.Path,
    *,
    filters: list[FilterSpec],
    group: str = "cp",
) -> None:
    """Apply a chain of filters to one group of a timeseries H5.

    Reads the source file, calls :func:`cfdmod.filters.apply_filters`
    on the loaded array, and writes the filtered series to
    ``output_h5`` with the same on-disk shape as the input -- preserving
    geometry and time axis, plus a ``/processing_metadata`` block.

    Args:
        input_h5: Source timeseries H5 (e.g. ``cp.default.time_series.h5``).
        output_h5: Destination timeseries H5. Overwritten if it exists.
        filters: Sequence of filter specs applied in order.
        group: Coefficient group name inside the H5 (e.g. ``"cp"``,
            ``"cf_x"``, ``"cm_y"``).

    Output layout:

    - ``/Triangles + /Geometry``: copied verbatim from ``input_h5``.
    - ``/meta/time_steps + /meta/time_normalized``: copied verbatim
      (filter preserves the time axis; only sample values change).
    - ``/{group}/t{T}``: the filtered series, one dataset per timestep.
    - ``/processing_metadata``: records the applied filter chain plus
      the input file path.
    - A temporal XDMF sibling next to ``output_h5`` so ParaView can
      animate the filtered field.
    """
    if not filters:
        raise ValueError("apply_filters_h5: filters list is empty (no-op)")
    output_h5 = pathlib.Path(output_h5)
    if output_h5.exists():
        output_h5.unlink()

    keys = get_pressure_keys(input_h5, group)
    if not keys:
        raise ValueError(f"{input_h5}:/{group} has no timesteps")

    with h5py.File(input_h5, "r") as f:
        triangles = f["Triangles"][:]
        vertices = f["Geometry"][:]
        data = np.stack([f[group][k][:].astype(np.float64) for _, k in keys])

    meta = read_timeseries_meta(input_h5)
    time_steps = np.asarray(meta["time_steps"], dtype=np.float64)
    time_normalized = np.asarray(meta["time_normalized"], dtype=np.float64)

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
        f"apply_filters_h5: {len(filters)} filter(s) on {input_h5} -> {output_h5} "
        f"(n_time={data.shape[0]}, n_tri={data.shape[1]}, dt={dt:.6g})"
    )

    out = apply_filters(data, dt, filters)

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
    logger.info(f"apply_filters_h5: wrote {output_h5}")
