"""v3 paradigm: data sources, ops, pipelines.

This package is the functional core of the v3 paradigm introduced by
issue #131. It holds value objects (:class:`DataSource`,
:class:`TimeAxis`, :class:`Topology`, :class:`Grouping`,
:class:`FieldMeta`), the protocols that gate the small-vs-large data
seam (:class:`FieldStore`, :class:`Storage`), and the algebra +
pipeline + container primitives every recipe is built from.

Concrete backends live under ``cfdmod.adapters``; ops and recipes
build on top of this package.

Phase 1 of the migration plan is the public surface below. The legacy
public API in ``cfdmod`` remains unchanged.
"""

from __future__ import annotations

from cfdmod.core.container import Container
from cfdmod.core.data_source import (
    DataSource,
    GroupsDataSource,
    ModesDataSource,
    PointsDataSource,
    SurfaceDataSource,
    VolumeDataSource,
)
from cfdmod.core.field_meta import FieldMeta
from cfdmod.core.grouping import Grouping, elements_in_group, groups_in
from cfdmod.core.pipeline import Pipeline, compose, identity
from cfdmod.core.protocols import FieldStore, Logger, Pool, Storage
from cfdmod.core.time_axis import TimeAxis
from cfdmod.core.topology import CellType, ElementMeta, Topology

__all__ = [
    "Container",
    "DataSource",
    "GroupsDataSource",
    "ModesDataSource",
    "PointsDataSource",
    "SurfaceDataSource",
    "VolumeDataSource",
    "FieldMeta",
    "Grouping",
    "groups_in",
    "elements_in_group",
    "Pipeline",
    "compose",
    "identity",
    "FieldStore",
    "Storage",
    "Logger",
    "Pool",
    "TimeAxis",
    "CellType",
    "ElementMeta",
    "Topology",
]
