"""Convenience helpers to read pressure-coefficient timeseries H5 files
into pandas DataFrames -- and from there to CSV or matplotlib.

The v2 timeseries layout stores one 1-D dataset per timestep under
``/{group}/t{T}``, with mesh metadata at ``/Triangles + /Geometry`` and
time arrays at ``/meta``. This module flattens that into the
spreadsheet-friendly **wide-form** ``pandas.DataFrame``::

    index    = time_normalized (float)
    columns  = triangle index (int) or region representative
    values   = scalar coefficient at (time, triangle)

For Cf and Cm, where every triangle in a region carries the same value,
``regions=True`` deduplicates by unique value to give one column per
region. For Cp (truly per-triangle), pass ``triangles=[...]`` to filter
to a tractable subset before exporting.
"""

from __future__ import annotations

__all__ = ["read_timeseries_df", "to_csv", "plot_timeseries"]

import pathlib
from collections.abc import Iterable
from typing import TYPE_CHECKING

import h5py
import numpy as np
import pandas as pd

if TYPE_CHECKING:
    from matplotlib.axes import Axes


def _read_meta(h5_path: pathlib.Path) -> dict[float, float]:
    """Return a {raw_time: time_normalized} map from the file's /meta group."""
    with h5py.File(h5_path, "r") as f:
        if "meta" not in f:
            return {}
        meta = f["meta"]
        return {
            float(t): float(tn) for t, tn in zip(meta["time_steps"][:], meta["time_normalized"][:])
        }


def read_timeseries_df(
    h5_path: pathlib.Path | str,
    group: str,
    *,
    triangles: Iterable[int] | None = None,
    regions: bool = False,
    timestep_range: tuple[float, float] | None = None,
    max_columns: int = 200,
) -> pd.DataFrame:
    """Read a coefficient timeseries from an XDMF+H5 into a wide-form DataFrame.

    Args:
        h5_path: Timeseries H5 path (e.g. ``cp.default.time_series.h5``,
            ``Cf.containers.pack.time_series.h5``).
        group: Coefficient group inside the file. Examples:
            ``"cp"`` (in a Cp file), ``"cf_x"`` / ``"cf_y"`` / ``"cf_z"``
            (in a Cf file), ``"cm_x"`` / ``"cm_y"`` / ``"cm_z"`` (Cm file).
        triangles: Optional list/iterable of triangle indices to keep as
            columns. Mutually exclusive with ``regions=True``.
        regions: When True, deduplicate columns by their value pattern --
            one representative triangle per unique value vector, named by
            the chosen triangle's integer index. Correct for Cf/Cm files
            where each region contributes a constant value to all its
            triangles. **Do not** use on per-triangle data (Cp).
        timestep_range: Optional ``(t_min, t_max)`` filter applied on the
            raw time keys (``t{T}`` keys, not on the normalized index).
        max_columns: Refuse to return a DataFrame wider than this many
            columns unless ``triangles`` or ``regions`` is set; protects
            callers that forget to filter on a per-triangle file
            (an 80k-tri Cp would be 80k columns wide, way past
            spreadsheet usability).

    Returns:
        DataFrame indexed by ``time_normalized`` with one column per
        retained triangle / region representative.

    Raises:
        ValueError: If the file has no ``/{group}`` group, the timestep
            filter yields no rows, both ``triangles`` and ``regions`` are
            requested, or the unfiltered column count exceeds
            ``max_columns``.
    """
    if triangles is not None and regions:
        raise ValueError("pass either triangles=... or regions=True, not both")

    h5_path = pathlib.Path(h5_path)
    raw_to_norm = _read_meta(h5_path)

    with h5py.File(h5_path, "r") as f:
        if group not in f:
            raise ValueError(f"{h5_path}:/{group} not found")
        grp = f[group]
        all_keys = sorted(grp.keys(), key=lambda k: float(k[1:]))
        raw_times = np.array([float(k[1:]) for k in all_keys])

        if timestep_range is not None:
            t_min, t_max = timestep_range
            mask = (raw_times >= t_min) & (raw_times <= t_max)
            keys = [k for k, m in zip(all_keys, mask) if m]
            raw_times = raw_times[mask]
        else:
            keys = all_keys

        if not keys:
            raise ValueError(f"no timesteps in {h5_path}:/{group} for range {timestep_range}")

        if triangles is not None:
            tri_idx = np.array(sorted(set(int(t) for t in triangles)))
            data = np.stack([grp[k][:][tri_idx] for k in keys])
            cols: list[int] = tri_idx.tolist()
        else:
            data = np.stack([grp[k][:] for k in keys])
            n_tri = data.shape[1]
            if regions:
                # Each region's triangles share their value vector across
                # time, so the first-timestep value already separates regions.
                # Pick the lowest-index representative per unique value.
                _, first_index = np.unique(data[0], return_index=True)
                tri_idx = np.sort(first_index)
                data = data[:, tri_idx]
                cols = tri_idx.tolist()
            else:
                if n_tri > max_columns:
                    raise ValueError(
                        f"{h5_path}:/{group} has {n_tri} columns -- too wide. "
                        "Pass triangles=[...] to filter, regions=True to "
                        "deduplicate per region (Cf/Cm), or override "
                        "max_columns explicitly."
                    )
                cols = list(range(n_tri))

    norm_index = pd.Index(
        [raw_to_norm.get(float(t), float(t)) for t in raw_times],
        name="time_normalized",
    )
    return pd.DataFrame(data, index=norm_index, columns=cols)


def to_csv(df: pd.DataFrame, path: pathlib.Path | str, **kwargs) -> None:
    """Save a timeseries DataFrame as CSV.

    Wide-form by default: first column is ``time_normalized`` (from the
    index), one column per retained triangle/region. Drops straight into
    Google Sheets / Excel via *Open* / *Import*.

    Extra keyword arguments are forwarded to :meth:`pandas.DataFrame.to_csv`.
    """
    df.to_csv(pathlib.Path(path), index=True, **kwargs)


def plot_timeseries(
    df: pd.DataFrame,
    columns: Iterable[int] | None = None,
    *,
    ax: "Axes | None" = None,
    title: str | None = None,
    ylabel: str = "value",
    **plot_kwargs,
) -> "Axes":
    """One-line matplotlib plot of a timeseries DataFrame.

    Args:
        df: A DataFrame returned by :func:`read_timeseries_df`.
        columns: Optional subset of columns (triangle indices) to plot;
            defaults to all columns currently in the DataFrame.
        ax: Existing :class:`matplotlib.axes.Axes` to draw into; a new
            figure is created if omitted.
        title: Plot title.
        ylabel: Y-axis label (default ``"value"`` -- override per
            coefficient, e.g. ``"Cp"`` / ``"Cf_x"``).
        **plot_kwargs: Extra args passed to ``DataFrame.plot``.

    Returns:
        The matplotlib Axes the plot was drawn into.
    """
    import matplotlib.pyplot as plt

    if columns is not None:
        df = df.loc[:, list(columns)]
    if ax is None:
        _, ax = plt.subplots()
    df.plot(ax=ax, **plot_kwargs)
    ax.set_xlabel("time_normalized")
    ax.set_ylabel(ylabel)
    if title is not None:
        ax.set_title(title)
    return ax
