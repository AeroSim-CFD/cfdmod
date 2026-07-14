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
from cfdmod.core.errors import (
    CfdmodError,
    OpError,
    StorageKeyError,
    TemplateError,
    TemplateReferenceError,
)
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
from cfdmod.core.freshness import (
    OutputStatus,
    output_status,
    signature,
)
from cfdmod.core.pipeline_yaml import (
    OP_REGISTRY,
    DigestStrategy,
    FreshnessConfig,
    OpInfo,
    PipelineTemplate,
    list_ops,
    load_template,
    op_info,
    register_op,
    run_template,
)
from cfdmod.core.protocols import BlobStore, FieldStore, Logger, Pool, Storage
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
    "BlobStore",
    "Logger",
    "Pool",
    "TimeAxis",
    "CellType",
    "ElementMeta",
    "Topology",
    "load_template",
    "run_template",
    "PipelineTemplate",
    "OP_REGISTRY",
    "register_op",
    "OpInfo",
    "list_ops",
    "op_info",
    "DigestStrategy",
    "FreshnessConfig",
    "OutputStatus",
    "output_status",
    "signature",
    "CfdmodError",
    "TemplateError",
    "TemplateReferenceError",
    "OpError",
    "StorageKeyError",
]
