"""Attach (or replace) a grouping on a data source.

Pure metadata update: no field arrays touched, no topology change. The
grouping is keyed by its ``name`` and stored under
``DataSource.groupings``.

This is the ``group-belonging index`` op from the proposal -- the
cheapest way to enrich a data source with a partition for downstream
ops (``filter_by_grouping``, ``field_series_for_groups``).
"""

from __future__ import annotations

__all__ = ["AttachGroupingParams", "attach_grouping"]

from typing import ClassVar, Literal

from cfdmod.core.data_source import DataSource
from cfdmod.core.grouping import Grouping
from cfdmod.core.ops import OpParams


class AttachGroupingParams(OpParams):
    """Parameters for :func:`attach_grouping`.

    Attributes:
        grouping: The grouping to attach. Its ``n_elements`` must match
            the data source's element axis.
    """

    kind: Literal["attach_grouping"] = "attach_grouping"
    grouping: Grouping

    chunkable_along: ClassVar[frozenset[str]] = frozenset({"time", "elements"})


def attach_grouping(ds: DataSource, p: AttachGroupingParams) -> DataSource:
    if p.grouping.n_elements != ds.n_elements:
        raise ValueError(
            f"grouping {p.grouping.name!r} has {p.grouping.n_elements} entries; "
            f"expected {ds.n_elements}"
        )
    return ds.with_grouping(p.grouping)
