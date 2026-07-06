"""Topology and element metadata for a :class:`DataSource`.

Two concerns live here:

- :class:`Topology` -- the connectivity + vertex coordinates of the
  underlying mesh (or the bare points of a points data source). The
  ``cell_type`` discriminator is in place from day one so volume
  export can be added later additively.
- :class:`ElementMeta` -- per-element scalar / vector attributes:
  position (centroid), area, volume, normal. Optional free-form metadata
  is allowed for "station name"-style annotations.

A :class:`GroupsDataSource` does *not* own its topology; it carries a
reference to the parent surface's topology plus a per-element group
index. That keeps groups data sources cheap and prevents the parent
mesh from drifting from the grouped view.
"""

from __future__ import annotations

__all__ = [
    "CellType",
    "Topology",
    "ElementMeta",
]

from typing import Annotated, Any, Literal

import numpy as np
from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

CellType = Literal["triangle", "point", "cell"]
"""Supported cell-type discriminators.

- ``triangle``: 3-node 2D triangular faces (the existing surface
  format). ``connectivity`` has shape ``(n_elements, 3)``.
- ``point``: bare points / probes. ``connectivity`` is empty
  (shape ``(0, 0)``).
- ``cell``: 3D volumetric cells. Reserved; not implemented in Phase 1
  but held by the discriminator so Phase-N volume export does not
  need a schema change.
"""


def _arr(value: Any, dtype: np.dtype) -> np.ndarray:
    """Coerce inputs to ``numpy.ndarray`` with the requested dtype.

    Pydantic does not natively validate ``numpy.ndarray`` fields; this
    helper plus the ``arbitrary_types_allowed=True`` config gives a
    consistent point of entry.
    """
    arr = np.asarray(value, dtype=dtype)
    return arr


class Topology(BaseModel):
    """Mesh connectivity + vertex coordinates for a data source.

    Frozen and immutable. Geometric ops (rigid-body transformation,
    rescale) produce a new :class:`Topology` rather than mutating in
    place.

    Attributes:
        cell_type: One of :data:`CellType`. Locks the connectivity
            schema.
        connectivity: For ``triangle`` -> ``(n_elements, 3)`` int32
            indices into ``vertices``. For ``point`` -> ``(0, 0)`` (no
            connectivity). For ``cell`` (reserved) -> implementation
            specific.
        vertices: ``(n_vertices, 3)`` float64 vertex coordinates.

    The shapes mirror the existing on-disk layout in
    ``cfdmod/io/xdmf.py`` exactly: ``/Triangles`` -> int32 (N, 3),
    ``/Geometry`` -> float64 (V, 3). No format change.
    """

    model_config = ConfigDict(frozen=True, arbitrary_types_allowed=True)

    cell_type: CellType
    connectivity: np.ndarray
    vertices: np.ndarray

    @field_validator("connectivity", mode="before")
    @classmethod
    def _coerce_connectivity(cls, v: Any) -> np.ndarray:
        return _arr(v, np.dtype("int32"))

    @field_validator("vertices", mode="before")
    @classmethod
    def _coerce_vertices(cls, v: Any) -> np.ndarray:
        return _arr(v, np.dtype("float64"))

    @model_validator(mode="after")
    def _check_shapes(self) -> "Topology":
        if self.vertices.ndim != 2 or self.vertices.shape[1] != 3:
            raise ValueError(
                f"vertices must have shape (n_vertices, 3); got {self.vertices.shape}"
            )
        if self.cell_type == "triangle":
            if self.connectivity.ndim != 2 or self.connectivity.shape[1] != 3:
                raise ValueError(
                    "triangle connectivity must have shape (n_elements, 3); got "
                    f"{self.connectivity.shape}"
                )
            if self.vertices.shape[0] > 0 and self.connectivity.size > 0:
                max_idx = int(self.connectivity.max())
                if max_idx >= self.vertices.shape[0]:
                    raise ValueError(
                        f"connectivity references vertex {max_idx} but only "
                        f"{self.vertices.shape[0]} vertices were provided"
                    )
        elif self.cell_type == "point":
            if self.connectivity.size != 0:
                raise ValueError(
                    "point topology must have empty connectivity; got "
                    f"shape {self.connectivity.shape}"
                )
        return self

    @property
    def n_elements(self) -> int:
        """Number of elements (triangles, points, or cells)."""
        if self.cell_type == "point":
            return self.vertices.shape[0]
        return self.connectivity.shape[0]

    @property
    def n_vertices(self) -> int:
        return self.vertices.shape[0]

    @classmethod
    def points(cls, vertices: Any) -> "Topology":
        """Build a points topology from a ``(n_points, 3)`` array."""
        verts = _arr(vertices, np.dtype("float64"))
        return cls(
            cell_type="point",
            connectivity=np.empty((0, 0), dtype=np.int32),
            vertices=verts,
        )

    @classmethod
    def triangles(cls, connectivity: Any, vertices: Any) -> "Topology":
        """Build a triangle topology from connectivity + vertex arrays."""
        return cls(
            cell_type="triangle",
            connectivity=connectivity,
            vertices=vertices,
        )


class ElementMeta(BaseModel):
    """Per-element scalar / vector attributes.

    Every column is optional; a points data source typically only sets
    ``position``, a surface data source typically sets ``position`` +
    ``area`` + ``normal``, a volume data source adds ``volume``.

    All array columns share the leading axis ``n_elements``. Cross-
    column shape checks happen at :class:`DataSource` construction
    time, not here, because the canonical length lives there.

    Free-form per-element metadata (e.g. station name) belongs in
    ``annotations``: a dict whose values are arrays of length
    ``n_elements`` or constants.
    """

    model_config = ConfigDict(frozen=True, arbitrary_types_allowed=True)

    position: np.ndarray | None = None
    area: np.ndarray | None = None
    volume: np.ndarray | None = None
    normal: np.ndarray | None = None
    annotations: Annotated[
        dict[str, Any],
        Field(default_factory=dict, description="Free-form per-element metadata."),
    ]

    @field_validator("position", "normal", mode="before")
    @classmethod
    def _coerce_vec3(cls, v: Any) -> np.ndarray | None:
        if v is None:
            return None
        return _arr(v, np.dtype("float64"))

    @field_validator("area", "volume", mode="before")
    @classmethod
    def _coerce_scalar(cls, v: Any) -> np.ndarray | None:
        if v is None:
            return None
        return _arr(v, np.dtype("float64"))

    @model_validator(mode="after")
    def _check_shapes(self) -> "ElementMeta":
        for name in ("position", "normal"):
            arr = getattr(self, name)
            if arr is not None and (arr.ndim != 2 or arr.shape[1] != 3):
                raise ValueError(f"{name} must have shape (n_elements, 3); got {arr.shape}")
        for name in ("area", "volume"):
            arr = getattr(self, name)
            if arr is not None and arr.ndim != 1:
                raise ValueError(f"{name} must have shape (n_elements,); got {arr.shape}")
        return self
