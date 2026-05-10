"""Unit tests for :class:`Topology` and :class:`ElementMeta`."""

from __future__ import annotations

import numpy as np
import pytest

from cfdmod.core import ElementMeta, Topology


def test_triangle_topology_validates_index_range():
    verts = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0]], dtype=np.float64)
    topo = Topology.triangles([[0, 1, 2]], verts)
    assert topo.cell_type == "triangle"
    assert topo.n_elements == 1
    assert topo.n_vertices == 3
    with pytest.raises(ValueError):
        Topology.triangles([[0, 1, 3]], verts)


def test_triangle_topology_rejects_non_3_columns():
    verts = np.zeros((3, 3))
    with pytest.raises(ValueError):
        Topology(cell_type="triangle", connectivity=[[0, 1]], vertices=verts)


def test_points_topology_has_empty_connectivity():
    verts = np.array([[0, 0, 0], [1, 0, 0]], dtype=np.float64)
    topo = Topology.points(verts)
    assert topo.cell_type == "point"
    assert topo.connectivity.size == 0
    assert topo.n_elements == 2


def test_element_meta_shape_validation():
    pos = np.zeros((3, 3))
    area = np.array([1.0, 2.0, 3.0])
    em = ElementMeta(position=pos, area=area)
    assert em.position.shape == (3, 3)
    assert em.area.shape == (3,)

    with pytest.raises(ValueError):
        ElementMeta(position=np.zeros((3, 2)))
    with pytest.raises(ValueError):
        ElementMeta(area=np.zeros((3, 1)))


def test_topology_is_frozen():
    topo = Topology.points(np.zeros((1, 3)))
    with pytest.raises(Exception):
        topo.cell_type = "triangle"  # type: ignore[misc]
