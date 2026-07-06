"""Frozen :class:`DataSource` value object plus the five concrete kinds.

The :class:`DataSource` is the unit of input and output for every op.
It binds together:

- a :class:`TimeAxis` (affine, never materialised);
- a :class:`Topology` (or ``None`` for purely tabular sources);
- an :class:`ElementMeta` (per-element scalars / vectors);
- a dict of :class:`Grouping` over the element axis;
- a :class:`FieldStore` (the small-vs-large-data seam) plus per-field
  :class:`FieldMeta`;
- a free-form ``attrs`` dict for source-level metadata.

Every method returns a *new* :class:`DataSource`. Field arrays inside
the underlying :class:`FieldStore` are shared by reference unless an
op explicitly rewrites them; large datasets do not duplicate.

Five concrete kinds are exposed; they are thin subclasses that lock
the ``kind`` discriminator and constrain which :class:`Topology`
``cell_type`` is admissible.
"""

from __future__ import annotations

__all__ = [
    "DataSource",
    "SurfaceDataSource",
    "VolumeDataSource",
    "PointsDataSource",
    "GroupsDataSource",
    "ModesDataSource",
]

from typing import Any, Literal

import numpy as np
from pydantic import BaseModel, ConfigDict, Field, model_validator

from cfdmod.core.field_meta import FieldMeta
from cfdmod.core.grouping import Grouping
from cfdmod.core.protocols import FieldStore
from cfdmod.core.time_axis import TimeAxis
from cfdmod.core.topology import ElementMeta, Topology

DataSourceKind = Literal["surface", "volume", "points", "groups", "modes"]


class DataSource(BaseModel):
    """Base frozen value object.

    Subclasses lock :attr:`kind` and (optionally) the admissible
    :attr:`Topology.cell_type`. Methods on this base never mutate;
    they always return a new instance via ``model_copy(update=...)``.

    Attributes:
        kind: One of ``surface``, ``volume``, ``points``, ``groups``,
            ``modes``. Locked by each subclass.
        time: Affine time axis. Time-aggregated outputs use
            ``n_timesteps == 0``.
        topology: Mesh connectivity / coordinates, when applicable.
            ``None`` is permitted for some kinds (notably ``modes``).
        elements: Per-element scalar / vector attributes.
        groupings: Mapping of grouping name -> :class:`Grouping`.
        fields: A :class:`FieldStore`. Carries the heavy arrays.
        field_meta: Mapping of field name -> :class:`FieldMeta`.
        attrs: Free-form source-level metadata.
    """

    model_config = ConfigDict(frozen=True, arbitrary_types_allowed=True)

    kind: DataSourceKind
    time: TimeAxis
    topology: Topology | None
    elements: ElementMeta
    groupings: dict[str, Grouping] = Field(default_factory=dict)
    fields: FieldStore
    field_meta: dict[str, FieldMeta] = Field(default_factory=dict)
    attrs: dict[str, Any] = Field(default_factory=dict)

    @model_validator(mode="after")
    def _check_consistency(self) -> "DataSource":
        n = self.n_elements
        for col in ("position", "area", "volume", "normal"):
            arr = getattr(self.elements, col)
            if arr is not None and arr.shape[0] != n:
                raise ValueError(
                    f"elements.{col} length {arr.shape[0]} does not match " f"n_elements={n}"
                )
        for gname, grouping in self.groupings.items():
            if grouping.n_elements != n:
                raise ValueError(
                    f"grouping {gname!r} has {grouping.n_elements} entries; " f"expected {n}"
                )
        for fname in self.fields.keys():
            shape = self.fields.shape(fname)
            if not shape or shape[0] != n:
                raise ValueError(
                    f"field {fname!r} leading axis {shape[0] if shape else None} "
                    f"does not match n_elements={n}"
                )
            t_axis_len = shape[1] if len(shape) > 1 else 0
            if self.time.is_time_aggregated and t_axis_len != 0:
                raise ValueError(
                    f"field {fname!r} has a time axis but the data source's "
                    "time axis is time-aggregated"
                )
            if not self.time.is_time_aggregated and t_axis_len != self.time.n_timesteps:
                raise ValueError(
                    f"field {fname!r} time axis length {t_axis_len} does not "
                    f"match TimeAxis.n_timesteps={self.time.n_timesteps}"
                )
        return self

    @property
    def n_elements(self) -> int:
        """Number of elements (rows) on this data source."""
        if self.topology is not None:
            return self.topology.n_elements
        if self.elements.position is not None:
            return int(self.elements.position.shape[0])
        # Fall back to whichever scalar column is present.
        for col in ("area", "volume"):
            arr = getattr(self.elements, col)
            if arr is not None:
                return int(arr.shape[0])
        # Nothing else available; ask the field store.
        for name in self.fields.keys():
            return int(self.fields.shape(name)[0])
        return 0

    @property
    def field_names(self) -> list[str]:
        return list(self.fields.keys())

    # ----- Functional updates -------------------------------------------------

    def _copy_validated(self, **update: Any) -> "DataSource":
        """``model_copy`` + re-run the consistency validators.

        Pydantic's ``model_copy(update=...)`` does *not* re-run validators,
        so a functional update could otherwise build a shape-inconsistent
        (frozen) DataSource silently. Re-validating keeps the invariants
        the frozen model advertises true after every update, not just at
        first construction.
        """
        updated = self.model_copy(update=update)
        return type(self).model_validate(dict(updated.__dict__))

    def with_time(self, new_time: TimeAxis) -> "DataSource":
        """Return a copy with a new :class:`TimeAxis`. Field shapes must
        already match the new axis -- this is a metadata update only."""
        return self._copy_validated(time=new_time)

    def with_topology(self, new_topology: Topology) -> "DataSource":
        return self._copy_validated(topology=new_topology)

    def with_elements(self, new_elements: ElementMeta) -> "DataSource":
        return self._copy_validated(elements=new_elements)

    def with_grouping(self, grouping: Grouping) -> "DataSource":
        """Add or replace a grouping. The grouping name is the key."""
        new_groupings = dict(self.groupings)
        new_groupings[grouping.name] = grouping
        return self._copy_validated(groupings=new_groupings)

    def without_grouping(self, name: str) -> "DataSource":
        new_groupings = {k: v for k, v in self.groupings.items() if k != name}
        return self.model_copy(update={"groupings": new_groupings})

    def with_field(
        self,
        name: str,
        value: np.ndarray,
        meta: FieldMeta | None = None,
    ) -> "DataSource":
        """Add or replace a field. The :class:`FieldStore` decides
        whether the array is shared by reference or copied."""
        new_store = self.fields.with_field(name, value)
        new_meta = dict(self.field_meta)
        new_meta[name] = meta or FieldMeta(name=name)
        return self._copy_validated(fields=new_store, field_meta=new_meta)

    def with_attrs(self, **updates: Any) -> "DataSource":
        new_attrs = dict(self.attrs)
        new_attrs.update(updates)
        return self.model_copy(update={"attrs": new_attrs})


# ---------------------------------------------------------------------------
# Concrete kinds
# ---------------------------------------------------------------------------


class SurfaceDataSource(DataSource):
    """Faces (2D triangular cells) with optional timesteps.

    Topology cell type must be ``triangle``. Mirrors the existing
    cfdmod XDMF+H5 timeseries layout: ``/Triangles``, ``/Geometry``,
    ``/{group}/t{T}``.
    """

    kind: Literal["surface"] = "surface"

    @model_validator(mode="after")
    def _check_surface(self) -> "SurfaceDataSource":
        if self.topology is None or self.topology.cell_type != "triangle":
            raise ValueError(
                "SurfaceDataSource requires a triangle Topology; got "
                f"{None if self.topology is None else self.topology.cell_type!r}"
            )
        return self


class VolumeDataSource(DataSource):
    """3D cells with optional timesteps.

    Topology cell type must be ``cell``. Reserved -- not a Phase 1
    target. The class is here so volume export can be added later
    additively rather than as a schema change.
    """

    kind: Literal["volume"] = "volume"

    @model_validator(mode="after")
    def _check_volume(self) -> "VolumeDataSource":
        if self.topology is None or self.topology.cell_type != "cell":
            raise ValueError(
                "VolumeDataSource requires a cell Topology; got "
                f"{None if self.topology is None else self.topology.cell_type!r}"
            )
        return self


class PointsDataSource(DataSource):
    """Bare points / probes / vertical profiles.

    Covers the existing :class:`InflowData` (probe array + per-component
    timeseries) and :class:`s1.profile.Profile` (1-D vertical profile,
    no time axis). Topology cell type is ``point``; connectivity is
    empty.
    """

    kind: Literal["points"] = "points"

    @model_validator(mode="after")
    def _check_points(self) -> "PointsDataSource":
        if self.topology is None or self.topology.cell_type != "point":
            raise ValueError(
                "PointsDataSource requires a point Topology; got "
                f"{None if self.topology is None else self.topology.cell_type!r}"
            )
        return self


class GroupsDataSource(DataSource):
    """One row per group: an aggregation over a parent surface.

    A groups data source carries fields whose leading axis is the
    *group* index, not the original element index. Its topology is
    *chained*: it borrows the parent surface's :class:`Topology` plus
    a :class:`Grouping` mapping each parent element to a group.

    The class does not own a triangulation of the groups themselves
    (each group is in general not a single triangle). This avoids the
    "non-triangular faces" trap.

    Attributes:
        parent_topology: The parent surface's triangle topology.
        parent_grouping: A :class:`Grouping` over the parent surface's
            elements that determines membership.
    """

    kind: Literal["groups"] = "groups"

    parent_topology: Topology
    parent_grouping: Grouping

    @model_validator(mode="after")
    def _check_groups(self) -> "GroupsDataSource":
        if self.parent_topology.cell_type != "triangle":
            raise ValueError(
                "GroupsDataSource.parent_topology must be a triangle topology; got "
                f"{self.parent_topology.cell_type!r}"
            )
        if self.parent_grouping.n_elements != self.parent_topology.n_elements:
            raise ValueError(
                "GroupsDataSource.parent_grouping must have one entry per parent element"
            )
        # GroupsDataSource has no independent topology of its own.
        if self.topology is not None:
            raise ValueError(
                "GroupsDataSource must not carry an independent topology; topology is "
                "chained to parent_topology + parent_grouping"
            )
        return self


class ModesDataSource(DataSource):
    """Modal axis: one row per mode, fields are generalised-displacement
    timeseries.

    No spatial topology; the original mesh / structural data lives
    alongside in the recipe context. ``elements`` typically carries an
    annotation column with mode labels.
    """

    kind: Literal["modes"] = "modes"

    @model_validator(mode="after")
    def _check_modes(self) -> "ModesDataSource":
        if self.topology is not None:
            raise ValueError(
                "ModesDataSource does not carry a topology; got " f"{self.topology.cell_type!r}"
            )
        return self
