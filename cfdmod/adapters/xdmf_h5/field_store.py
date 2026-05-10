"""H5-backed :class:`FieldStore`.

Reads and writes per-timestep arrays from an h5 file using the existing
``/{group}/t{T}`` layout. Each field on the data source corresponds to
one h5 group whose children are the per-timestep datasets.

The store keeps the file *closed* between calls. Each :meth:`read`
opens the file in ``r`` mode and slices the requested slab, so multi-
process reads work without locking dance. For very hot loops, an
explicit ``with field_store.open() as ...`` could be added later (not
needed for the Phase 1/2 round-trip targets).

Time-aggregated fields are accommodated by representing them as a
single dataset at ``/{group}/{stat_name}`` (no ``t{T}`` indirection).
The :class:`XdmfH5Storage` decides which mode applies based on the
:class:`DataSource`'s :class:`TimeAxis`.
"""

from __future__ import annotations

__all__ = ["H5FieldStore"]

import pathlib
from typing import Iterable

import h5py
import numpy as np

from cfdmod.adapters.memory.field_store import MemoryFieldStore


class H5FieldStore:
    """Per-:class:`DataSource` field store backed by an h5 file.

    Args:
        h5_path: Path to the h5 file.
        field_groups: Mapping of field name -> h5 group path (no
            leading slash). For a Cp file: ``{"cp": "cp"}``. For a Cf
            file: ``{"cf_x": "cf_x", "cf_y": "cf_y", "cf_z": "cf_z"}``.
            For a stats file: ``{"mean": "<grp>/mean"}`` (a path to a
            single dataset rather than a t-keyed group).
        time_keys: Sorted list of t-keyed dataset names (e.g.
            ``["t0.0", "t0.1", ...]``). The same time axis is assumed
            across every group; this matches every existing cfdmod
            timeseries file. Pass an empty list for time-aggregated
            stores.
        n_elements: Element-axis length. Cached so :meth:`shape` does
            not have to re-open the file.
        time_aggregated: When True, every field group is a *single
            dataset path* rather than a t-keyed group. Used for
            stats files.
    """

    __slots__ = (
        "_h5_path",
        "_field_groups",
        "_time_keys",
        "_n_elements",
        "_time_aggregated",
        "_overlay",
    )

    def __init__(
        self,
        h5_path: pathlib.Path,
        field_groups: dict[str, str],
        *,
        time_keys: list[str],
        n_elements: int,
        time_aggregated: bool = False,
        overlay: MemoryFieldStore | None = None,
    ) -> None:
        self._h5_path = pathlib.Path(h5_path)
        self._field_groups = dict(field_groups)
        self._time_keys = list(time_keys)
        self._n_elements = int(n_elements)
        self._time_aggregated = bool(time_aggregated)
        # Functional updates land in the overlay rather than mutating the file.
        self._overlay = overlay if overlay is not None else MemoryFieldStore()

    # --- Inspection --------------------------------------------------------

    def keys(self) -> Iterable[str]:
        seen = set(self._overlay.keys())
        for k in self._field_groups.keys():
            if k not in seen:
                seen.add(k)
        return list(seen)

    def shape(self, name: str) -> tuple[int, ...]:
        if name in list(self._overlay.keys()):
            return self._overlay.shape(name)
        if name not in self._field_groups:
            raise KeyError(f"H5FieldStore has no field {name!r}")
        if self._time_aggregated:
            return (self._n_elements,)
        return (self._n_elements, len(self._time_keys))

    def dtype(self, name: str):
        if name in list(self._overlay.keys()):
            return self._overlay.dtype(name)
        return np.dtype("float64")

    # --- Read --------------------------------------------------------------

    def read(
        self,
        name: str,
        *,
        time_slice: slice | None = None,
        element_slice: slice | None = None,
        elements: np.ndarray | None = None,
    ) -> np.ndarray:
        if name in list(self._overlay.keys()):
            return self._overlay.read(
                name,
                time_slice=time_slice,
                element_slice=element_slice,
                elements=elements,
            )
        if name not in self._field_groups:
            raise KeyError(f"H5FieldStore has no field {name!r}")
        if element_slice is not None and elements is not None:
            raise ValueError("Pass either element_slice or elements, not both")

        group_path = self._field_groups[name]
        with h5py.File(self._h5_path, "r") as f:
            if self._time_aggregated:
                arr = f[group_path][:]
                if elements is not None:
                    return np.asarray(arr[elements])
                if element_slice is not None:
                    return np.asarray(arr[element_slice])
                return np.asarray(arr)

            grp = f[group_path]
            keys = self._time_keys if time_slice is None else self._time_keys[time_slice]
            if not keys:
                return np.empty((self._n_elements, 0), dtype=np.float64)

            # Read each timestep slab; column-stack into (n_elements, T).
            cols: list[np.ndarray] = []
            for k in keys:
                col = grp[k][:]
                if elements is not None:
                    col = col[elements]
                elif element_slice is not None:
                    col = col[element_slice]
                cols.append(col)
            return np.stack(cols, axis=1)

    # --- Write -------------------------------------------------------------

    def write(
        self,
        name: str,
        value: np.ndarray,
        *,
        time_slice: slice | None = None,
        element_slice: slice | None = None,
    ) -> None:
        # Functional-store policy: writes go to the overlay, never the
        # underlying file. The Storage adapter is the only place that
        # serialises an entire DataSource back to disk.
        self._overlay.write(
            name,
            value,
            time_slice=time_slice,
            element_slice=element_slice,
        )

    def with_field(self, name: str, value: np.ndarray) -> "H5FieldStore":
        new_overlay = self._overlay.with_field(name, value)
        return H5FieldStore(
            h5_path=self._h5_path,
            field_groups=self._field_groups,
            time_keys=self._time_keys,
            n_elements=self._n_elements,
            time_aggregated=self._time_aggregated,
            overlay=new_overlay,
        )

    # --- Adapter-internal accessors used by XdmfH5Storage on writeback ----

    @property
    def h5_path(self) -> pathlib.Path:
        return self._h5_path

    @property
    def field_groups(self) -> dict[str, str]:
        return dict(self._field_groups)

    @property
    def time_keys(self) -> list[str]:
        return list(self._time_keys)

    @property
    def overlay(self) -> MemoryFieldStore:
        return self._overlay
