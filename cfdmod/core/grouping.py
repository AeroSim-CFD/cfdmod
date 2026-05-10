"""Per-:class:`DataSource` element grouping.

Elements may belong to *one* group per grouping. A data source can
carry many groupings simultaneously (surface name, planar selection,
S1 separation, ...); each lives at a key under
``DataSource.groupings``.

This module provides:

- :class:`Grouping` -- a frozen view over the per-element index plus an
  optional ``id_to_label`` mapping for human-readable group names.
- helpers (:func:`groups_in`, :func:`elements_in_group`) shared by ops.

The :class:`GroupsDataSource` (in ``data_source.py``) is a *separate*
concept: it carries one row per group rather than one row per element,
and stores fields that are aggregations over the original elements.
"""

from __future__ import annotations

__all__ = [
    "Grouping",
    "groups_in",
    "elements_in_group",
]

from typing import Any

import numpy as np
from pydantic import BaseModel, ConfigDict, field_validator


class Grouping(BaseModel):
    """One grouping over a data source's element axis.

    Each element gets exactly one group id under this grouping; the
    sentinel ``-1`` marks "ungrouped". Groups are addressed by integer
    id; an optional :attr:`id_to_label` maps id to a human label
    (mirroring the legacy ``"{idx}-{body}"`` convention used in
    cfdmod's pressure pipeline).

    Attributes:
        name: Grouping name (e.g. ``"surface"``, ``"zoning"``).
        indices: ``(n_elements,)`` int32 array of group ids. ``-1`` =
            ungrouped.
        id_to_label: Optional dict mapping group id -> string label.
    """

    model_config = ConfigDict(frozen=True, arbitrary_types_allowed=True)

    name: str
    indices: np.ndarray
    id_to_label: dict[int, str] | None = None

    @field_validator("indices", mode="before")
    @classmethod
    def _coerce(cls, v: Any) -> np.ndarray:
        arr = np.asarray(v, dtype=np.int32)
        if arr.ndim != 1:
            raise ValueError(f"Grouping indices must be 1-D; got shape {arr.shape}")
        return arr

    @property
    def n_elements(self) -> int:
        return int(self.indices.shape[0])

    def label(self, group_id: int) -> str:
        """Resolve a group id to a label.

        Falls back to ``str(group_id)`` if no mapping is registered or
        the id is missing from the mapping.
        """
        if self.id_to_label is None:
            return str(group_id)
        return self.id_to_label.get(int(group_id), str(group_id))


def groups_in(grouping: Grouping, *, include_ungrouped: bool = False) -> np.ndarray:
    """Sorted unique group ids in a grouping.

    Args:
        grouping: The grouping to inspect.
        include_ungrouped: If False (default), ``-1`` is dropped from
            the result.
    """
    uniq = np.unique(grouping.indices)
    if not include_ungrouped:
        uniq = uniq[uniq != -1]
    return uniq


def elements_in_group(grouping: Grouping, group_id: int) -> np.ndarray:
    """Indices of every element belonging to ``group_id``."""
    return np.flatnonzero(grouping.indices == int(group_id))
