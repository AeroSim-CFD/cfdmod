"""Debug helpers for files on disk.

Functions here are meant for interactive use (notebook / REPL) when you
need to peek at an HDF5 file's structure or pull all timesteps from a
group into a single array. None of them are part of the pipeline; they
exist purely to make ad-hoc inspection painless.

Two helpers ship today:

- :func:`inspect_h5` -- print a human-readable tree of an HDF5 file
  (groups, datasets, shapes, dtypes, attributes). Groups whose children
  all match the ``t{T}`` timestep convention used by the pipeline are
  collapsed to a one-line summary so a 10k-timestep file does not flood
  the screen.
- :func:`read_all_timesteps` -- read every ``t{T}`` dataset under a
  group into one stacked array, sorted by time. Convenient for ad-hoc
  plotting and quick stats; for production streaming, prefer the
  per-step API in :mod:`cfdmod.io.xdmf`.
"""

from __future__ import annotations

__all__ = ["inspect_h5", "read_all_timesteps"]

import pathlib
import re

import h5py
import numpy as np

_TIME_KEY_RE = re.compile(r"^t-?\d+(\.\d+)?$")


def inspect_h5(
    path: pathlib.Path | str,
    *,
    show_attrs: bool = True,
    show_meta_values: bool = True,
) -> None:
    """Print a human-readable tree of an HDF5 file's contents.

    Walks every group and dataset; reports shape and dtype for
    datasets and (optionally) HDF5 attributes for both groups and
    datasets. Groups whose children all match the ``t{T}`` timestep
    convention used by the pipeline are collapsed to a one-line
    summary (count, time range, per-step shape).

    Args:
        path: Path to an HDF5 file.
        show_attrs: When True, print HDF5 attributes per group / dataset
            as ``@key = value`` lines indented under their owner.
        show_meta_values: When True, print a short range/preview for
            small 1D datasets directly under ``/meta``. Avoids dumping
            large arrays while still showing the time axis at a glance.
    """
    path = pathlib.Path(path)
    print(f"{path.name}  ({path.stat().st_size:,} bytes)")
    with h5py.File(path, "r") as f:
        _print_attrs(f, indent=1, show_attrs=show_attrs)
        _walk(f, indent=1, show_attrs=show_attrs, show_meta_values=show_meta_values)


def read_all_timesteps(
    h5_path: pathlib.Path | str,
    group: str,
) -> tuple[np.ndarray, np.ndarray]:
    """Read every ``t{T}`` dataset under ``/group`` into one stacked array.

    Sorts by time. Useful for dumping a whole timeseries into memory
    for plotting or quick statistics during debugging; for production
    streaming, prefer the per-step API in :mod:`cfdmod.io.xdmf`.

    Args:
        h5_path: Path to the timeseries H5 file.
        group: Name of the group holding ``t{T}`` datasets (e.g.
            ``"cp"``, ``"cf_x"``).

    Returns:
        ``(times, data)`` where ``times`` is a sorted ``float64``
        array of length ``n_timesteps`` and ``data`` is shape
        ``(n_timesteps, *single_step_shape)`` matching the per-step
        dtype.

    Raises:
        KeyError: If ``group`` does not exist in the file.
        ValueError: If ``group`` has no ``t{T}``-named children.
    """
    h5_path = pathlib.Path(h5_path)
    with h5py.File(h5_path, "r") as f:
        if group not in f:
            raise KeyError(f"group {group!r} not found in {h5_path}")
        grp = f[group]
        keys = [k for k in grp.keys() if _TIME_KEY_RE.match(k)]
        if not keys:
            raise ValueError(
                f"group {group!r} in {h5_path} has no t{{T}}-named children"
            )
        pairs = sorted((float(k[1:]), k) for k in keys)
        times = np.array([t for t, _ in pairs], dtype=np.float64)
        first = grp[pairs[0][1]]
        out = np.empty((len(pairs),) + first.shape, dtype=first.dtype)
        for i, (_, k) in enumerate(pairs):
            out[i] = grp[k][:]
    return times, out


def _walk(grp, indent: int, *, show_attrs: bool, show_meta_values: bool) -> None:
    pad = "  " * indent
    for name, item in grp.items():
        if isinstance(item, h5py.Dataset):
            line = f"{pad}{name}  shape={tuple(item.shape)} dtype={item.dtype}"
            preview = _maybe_preview(grp.name, item, show_meta_values)
            if preview:
                line += f"  {preview}"
            print(line)
            _print_attrs(item, indent=indent + 1, show_attrs=show_attrs)
        elif isinstance(item, h5py.Group):
            if _is_timestep_group(item):
                print(f"{pad}{name}/  {_summarise_timestep_group(item)}")
                _print_attrs(item, indent=indent + 1, show_attrs=show_attrs)
            else:
                print(f"{pad}{name}/")
                _print_attrs(item, indent=indent + 1, show_attrs=show_attrs)
                _walk(
                    item,
                    indent + 1,
                    show_attrs=show_attrs,
                    show_meta_values=show_meta_values,
                )


def _print_attrs(obj, indent: int, *, show_attrs: bool) -> None:
    if not show_attrs or len(obj.attrs) == 0:
        return
    pad = "  " * indent
    for k in obj.attrs.keys():
        v = obj.attrs[k]
        if isinstance(v, bytes):
            v = v.decode(errors="replace")
        v_repr = repr(v)
        if len(v_repr) > 80:
            v_repr = v_repr[:77] + "..."
        print(f"{pad}@{k} = {v_repr}")


def _is_timestep_group(grp) -> bool:
    if len(grp) == 0:
        return False
    for k in grp.keys():
        if not _TIME_KEY_RE.match(k):
            return False
        if not isinstance(grp[k], h5py.Dataset):
            return False
    return True


def _summarise_timestep_group(grp) -> str:
    times = sorted(float(k[1:]) for k in grp.keys())
    sample_key = next(iter(grp.keys()))
    sample = grp[sample_key]
    return (
        f"[{len(times)} timesteps, "
        f"t in [{times[0]:.6g}, {times[-1]:.6g}], "
        f"step shape={tuple(sample.shape)} dtype={sample.dtype}]"
    )


def _maybe_preview(group_name: str, dset: h5py.Dataset, show_meta_values: bool) -> str:
    if not show_meta_values:
        return ""
    if not (group_name == "/meta" or group_name.endswith("/meta")):
        return ""
    if dset.ndim != 1 or dset.size == 0:
        return ""
    arr = dset[:]
    if np.issubdtype(arr.dtype, np.number):
        return f"range=[{arr.min():.6g} .. {arr.max():.6g}]"
    if arr.dtype.kind in ("S", "O", "U"):
        sample = [s.decode() if isinstance(s, bytes) else str(s) for s in arr[:5]]
        tail = ", ..." if len(arr) > 5 else ""
        return f"first={', '.join(repr(s) for s in sample)}{tail}"
    return ""
