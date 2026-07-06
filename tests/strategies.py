"""Reusable hypothesis strategies for the v3 data-source layer (issue #141).

The :class:`~cfdmod.core.data_source.DataSource` paradigm is a good fit for
property-based testing: it is a small frozen value object with explicit
invariants (``_check_consistency``), pure functional updates, and a storage
round-trip whose contract is literally "read == write". These strategies
generate *always-consistent* sources so a drawn value already passes every
model validator at construction; a test then asserts an invariant over many
such draws rather than a single hand-written shape.

Design notes:

- Every generated array is finite ``float64`` by default. Numpy edge values
  (NaN, inf) are opt-in per strategy via ``allow_non_finite=True`` so we keep
  "does the math hold" separate from "how do we treat non-finite inputs" (the
  latter is its own decision -- see the #141 notes).
- Array magnitudes are bounded (``|x| <= 1e6``) so float formatting on the
  XDMF/H5 round-trip and ``np.allclose`` comparisons stay meaningful.
- Field names are simple ASCII identifiers with no ``/`` and never collide with
  the reserved h5 root datasets, so a surface/points source also round-trips
  through :class:`~cfdmod.adapters.xdmf_h5.XdmfH5Storage` (which maps each field
  to a top-level h5 group).
"""

from __future__ import annotations

__all__ = [
    "finite_floats",
    "field_arrays",
    "time_axes",
    "element_meta",
    "groupings",
    "triangle_topologies",
    "surface_data_sources",
    "points_data_sources",
    "groups_data_sources",
    "data_sources",
    "roundtrippable_data_sources",
]

import numpy as np
from hypothesis import strategies as st
from hypothesis.extra import numpy as hnp

from cfdmod.adapters.memory import MemoryFieldStore
from cfdmod.core.data_source import (
    GroupsDataSource,
    PointsDataSource,
    SurfaceDataSource,
)
from cfdmod.core.grouping import Grouping
from cfdmod.core.time_axis import TimeAxis
from cfdmod.core.topology import ElementMeta, Topology

# Small bounds keep example generation fast and the search space legible.
MAX_ELEMENTS = 6
MAX_TIMESTEPS = 5
MAX_FIELDS = 3
MAX_VERTICES = 8

# Reserved h5 root datasets/groups; a field named like one of these would not
# round-trip through the XDMF/H5 adapter, so field-name strategies avoid them.
_RESERVED_FIELD_NAMES = frozenset({"Triangles", "Geometry", "Connectivity", "meta", "stats"})


def finite_floats(*, allow_non_finite: bool = False):
    """A float strategy: finite and bounded by default, NaN/inf opt-in."""
    if allow_non_finite:
        return st.floats(allow_nan=True, allow_infinity=True, width=64)
    return st.floats(
        min_value=-1e6,
        max_value=1e6,
        allow_nan=False,
        allow_infinity=False,
        width=64,
    )


def _float_array(shape: tuple[int, ...], *, allow_non_finite: bool = False):
    """An ``hnp.arrays`` strategy of the given shape, float64."""
    return hnp.arrays(
        dtype=np.float64,
        shape=shape,
        elements=finite_floats(allow_non_finite=allow_non_finite),
    )


def field_arrays(n_elements: int, n_timesteps: int, *, allow_non_finite: bool = False):
    """Field arrays matching a source's shape contract.

    Shape is ``(n_elements,)`` for a time-aggregated source (``n_timesteps ==
    0``) and ``(n_elements, n_timesteps)`` otherwise -- exactly what
    ``_check_consistency`` requires.
    """
    shape: tuple[int, ...] = (n_elements,) if n_timesteps == 0 else (n_elements, n_timesteps)
    return _float_array(shape, allow_non_finite=allow_non_finite)


def _field_names(min_size: int = 1, max_size: int = MAX_FIELDS):
    """A set of distinct, round-trip-safe field names."""
    ident = st.text(
        alphabet=st.characters(min_codepoint=ord("a"), max_codepoint=ord("z")),
        min_size=1,
        max_size=6,
    ).filter(lambda s: s not in _RESERVED_FIELD_NAMES)
    return st.lists(ident, min_size=min_size, max_size=max_size, unique=True)


@st.composite
def time_axes(draw, *, aggregated: bool | None = None) -> TimeAxis:
    """A valid :class:`TimeAxis`.

    ``aggregated=True`` forces the no-time-axis form (``n_timesteps == 0``);
    ``aggregated=False`` forces a real time axis; ``None`` draws either.
    """
    if aggregated is None:
        aggregated = draw(st.booleans())
    initial = draw(st.floats(min_value=-1e4, max_value=1e4, allow_nan=False, allow_infinity=False))
    if aggregated:
        return TimeAxis(initial_time=initial, timestep_size=0.0, n_timesteps=0)
    n = draw(st.integers(min_value=1, max_value=MAX_TIMESTEPS))
    dt = draw(st.floats(min_value=1e-3, max_value=1e3, allow_nan=False, allow_infinity=False))
    return TimeAxis(initial_time=initial, timestep_size=dt, n_timesteps=n)


@st.composite
def element_meta(draw, n_elements: int, *, positive_area: bool = True) -> ElementMeta:
    """Per-element metadata with a consistent leading axis.

    ``position`` is always set (so a topology-less source can still resolve
    ``n_elements``); ``area`` is strictly positive by default so
    ``area_weighted_mean`` is well defined.
    """
    position = draw(_float_array((n_elements, 3)))
    if positive_area:
        area = draw(
            hnp.arrays(
                dtype=np.float64,
                shape=(n_elements,),
                elements=st.floats(
                    min_value=1e-3, max_value=1e3, allow_nan=False, allow_infinity=False
                ),
            )
        )
    else:
        area = draw(_float_array((n_elements,)))
    return ElementMeta(position=position, area=area)


@st.composite
def groupings(draw, n_elements: int, *, name: str | None = None) -> Grouping:
    """A :class:`Grouping` over ``n_elements`` elements.

    Group ids range over ``[-1, n_elements]`` where ``-1`` marks ungrouped.
    """
    if name is None:
        name = draw(
            st.text(
                alphabet=st.characters(min_codepoint=ord("a"), max_codepoint=ord("z")),
                min_size=1,
                max_size=6,
            )
        )
    indices = draw(
        hnp.arrays(
            dtype=np.int32,
            shape=(n_elements,),
            elements=st.integers(min_value=-1, max_value=n_elements),
        )
    )
    return Grouping(name=name, indices=indices)


@st.composite
def triangle_topologies(draw, *, n_elements: int | None = None) -> Topology:
    """A valid triangle :class:`Topology` with in-range connectivity."""
    n_vertices = draw(st.integers(min_value=3, max_value=MAX_VERTICES))
    vertices = draw(_float_array((n_vertices, 3)))
    if n_elements is None:
        n_elements = draw(st.integers(min_value=1, max_value=MAX_ELEMENTS))
    connectivity = draw(
        hnp.arrays(
            dtype=np.int32,
            shape=(n_elements, 3),
            elements=st.integers(min_value=0, max_value=n_vertices - 1),
        )
    )
    return Topology.triangles(connectivity, vertices)


@st.composite
def surface_data_sources(draw, *, allow_non_finite: bool = False) -> SurfaceDataSource:
    """A consistent :class:`SurfaceDataSource` (triangle topology + fields)."""
    topology = draw(triangle_topologies())
    n = topology.n_elements
    time = draw(time_axes())
    names = draw(_field_names())
    arrays = {
        name: draw(field_arrays(n, time.n_timesteps, allow_non_finite=allow_non_finite))
        for name in names
    }
    return SurfaceDataSource(
        time=time,
        topology=topology,
        elements=draw(element_meta(n)),
        fields=MemoryFieldStore(arrays),
    )


@st.composite
def points_data_sources(draw, *, allow_non_finite: bool = False) -> PointsDataSource:
    """A consistent :class:`PointsDataSource` (point topology + fields)."""
    n = draw(st.integers(min_value=1, max_value=MAX_ELEMENTS))
    vertices = draw(_float_array((n, 3)))
    time = draw(time_axes())
    names = draw(_field_names())
    arrays = {
        name: draw(field_arrays(n, time.n_timesteps, allow_non_finite=allow_non_finite))
        for name in names
    }
    return PointsDataSource(
        time=time,
        topology=Topology.points(vertices),
        elements=draw(element_meta(n)),
        fields=MemoryFieldStore(arrays),
    )


@st.composite
def groups_data_sources(draw, *, allow_non_finite: bool = False) -> GroupsDataSource:
    """A consistent :class:`GroupsDataSource`.

    Carries one row per group; the topology is chained to an independently
    drawn parent triangle surface plus a grouping over the parent elements.
    """
    n_groups = draw(st.integers(min_value=1, max_value=MAX_ELEMENTS))
    parent_topology = draw(triangle_topologies())
    parent_grouping = draw(groupings(parent_topology.n_elements, name="parent"))
    time = draw(time_axes())
    names = draw(_field_names())
    arrays = {
        name: draw(field_arrays(n_groups, time.n_timesteps, allow_non_finite=allow_non_finite))
        for name in names
    }
    return GroupsDataSource(
        time=time,
        topology=None,
        elements=draw(element_meta(n_groups)),
        fields=MemoryFieldStore(arrays),
        parent_topology=parent_topology,
        parent_grouping=parent_grouping,
    )


def data_sources(*, allow_non_finite: bool = False):
    """Any of the phase-1 concrete kinds (surface / points / groups)."""
    return st.one_of(
        surface_data_sources(allow_non_finite=allow_non_finite),
        points_data_sources(allow_non_finite=allow_non_finite),
        groups_data_sources(allow_non_finite=allow_non_finite),
    )


def roundtrippable_data_sources(*, allow_non_finite: bool = False):
    """Kinds the XDMF/H5 adapter reads back as themselves (surface / points).

    A :class:`GroupsDataSource` is intentionally excluded: the adapter
    broadcasts it onto its parent surface on write, so its storage contract is
    not "read == write" and it does not belong in a round-trip invariant.
    """
    return st.one_of(
        surface_data_sources(allow_non_finite=allow_non_finite),
        points_data_sources(allow_non_finite=allow_non_finite),
    )
