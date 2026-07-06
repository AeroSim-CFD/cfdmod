"""In-RAM :class:`FieldStore`.

Holds field arrays in a regular Python dict. Trivial implementation
that satisfies the :class:`cfdmod.core.protocols.FieldStore` protocol;
used by every unit test and by notebooks running on synthesised data.

Functional updates (:meth:`with_field`) build a new
:class:`MemoryFieldStore` and share unmodified arrays *by reference*
with the predecessor. Only the named field is replaced.
"""

from __future__ import annotations

__all__ = ["MemoryFieldStore"]

from typing import Iterable

import numpy as np


class MemoryFieldStore:
    """Plain dict-backed :class:`FieldStore`.

    Args:
        arrays: Mapping of field name -> ``numpy.ndarray``. The arrays
            are kept by reference (not copied); callers must not mutate
            them in place after handing them over. Use
            :meth:`with_field` to replace.
    """

    __slots__ = ("_arrays",)

    def __init__(self, arrays: dict[str, np.ndarray] | None = None) -> None:
        self._arrays: dict[str, np.ndarray] = {} if arrays is None else dict(arrays)

    def keys(self) -> Iterable[str]:
        return self._arrays.keys()

    def shape(self, name: str) -> tuple[int, ...]:
        return self._arrays[name].shape

    def dtype(self, name: str):
        return self._arrays[name].dtype

    def read(
        self,
        name: str,
        *,
        time_slice: slice | None = None,
        element_slice: slice | None = None,
        elements: np.ndarray | None = None,
    ) -> np.ndarray:
        if element_slice is not None and elements is not None:
            raise ValueError("Pass either element_slice or elements, not both")
        arr = self._arrays[name]
        if elements is not None:
            arr = arr[elements]
        elif element_slice is not None:
            arr = arr[element_slice]
        if time_slice is not None and arr.ndim == 2:
            arr = arr[:, time_slice]
        return arr

    def write(
        self,
        name: str,
        value: np.ndarray,
        *,
        time_slice: slice | None = None,
        element_slice: slice | None = None,
    ) -> None:
        if name not in self._arrays:
            if time_slice is not None or element_slice is not None:
                raise KeyError(
                    f"cannot partially write field {name!r}: it does not exist yet. "
                    "Create it first via with_field(name, full_array)."
                )
            self._arrays[name] = np.asarray(value)
            return
        target = self._arrays[name]
        if element_slice is None and time_slice is None:
            self._arrays[name] = np.asarray(value)
            return
        # Partial in-place write -- we own the array.
        e_slice = slice(None) if element_slice is None else element_slice
        if target.ndim == 2:
            t_slice = slice(None) if time_slice is None else time_slice
            target[e_slice, t_slice] = value
        else:
            if time_slice is not None:
                raise ValueError("time_slice not applicable to a 1-D field")
            target[e_slice] = value

    def with_field(self, name: str, value: np.ndarray) -> "MemoryFieldStore":
        new_arrays = dict(self._arrays)
        new_arrays[name] = np.asarray(value)
        return MemoryFieldStore(new_arrays)
