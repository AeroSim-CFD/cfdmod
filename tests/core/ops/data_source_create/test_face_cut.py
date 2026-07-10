"""Unit tests for the face_cut op.

face_cut geometrically slices a surface's triangles along axis-aligned
planes so a triangle straddling a boundary contributes its real partial
area to each side -- the exact counterpart to centroid ``zoning_grouping``.
"""

from __future__ import annotations

import numpy as np
import pytest

from cfdmod.adapters.memory import MemoryFieldStore
from cfdmod.core import (
    ElementMeta,
    SurfaceDataSource,
    TimeAxis,
    Topology,
)
from cfdmod.core.ops.data_source_create import FaceCutParams, face_cut
from cfdmod.core.ops.field import ForceContributionParams, force_contribution
from cfdmod.geometry.triangle_slicing import slice_triangles_with_parents


def _wall(cp_per_tri: np.ndarray, n_timesteps: int = 3) -> SurfaceDataSource:
    """A vertical wall (normal -y) spanning x in [0,2], z in [0,2].

    Two triangles, both straddling z = 1 so a z-cut actually slices them.
    ``cp_per_tri`` is one scalar per triangle, broadcast over time.
    """
    verts = np.array(
        [[0, 0, 0], [2, 0, 0], [2, 0, 2], [0, 0, 2]],
        dtype=np.float64,
    )
    tris = np.array([[0, 1, 2], [0, 2, 3]], dtype=np.int32)
    # Both triangles lie in the y=0 plane; outward normal is -y.
    normals = np.array([[0.0, -1.0, 0.0], [0.0, -1.0, 0.0]])
    cp = np.repeat(cp_per_tri[:, None], n_timesteps, axis=1).astype(np.float64)
    return SurfaceDataSource(
        time=TimeAxis(initial_time=0.0, timestep_size=1.0, n_timesteps=n_timesteps),
        topology=Topology.triangles(tris, verts),
        elements=ElementMeta(
            area=np.array([2.0, 2.0]),  # each triangle is half of the 2x2 wall
            normal=normals,
        ),
        fields=MemoryFieldStore({"cp": cp}),
    )


def _parent_area(ds: SurfaceDataSource) -> np.ndarray:
    tv = ds.topology.vertices[ds.topology.connectivity]
    e1 = tv[:, 1] - tv[:, 0]
    e2 = tv[:, 2] - tv[:, 0]
    return 0.5 * np.linalg.norm(np.cross(e1, e2), axis=1)


def test_face_cut_partitions_area_exactly():
    ds = _wall(np.array([1.0, 1.0]))
    out = face_cut(ds, FaceCutParams(z_intervals=[0.0, 1.0, 2.0], name="floor"))

    # Slicing actually happened (more fragments than parents).
    assert out.n_elements > ds.n_elements
    # Fragment areas partition the parent wall exactly (no duplication, no loss).
    assert out.elements.area.sum() == pytest.approx(_parent_area(ds).sum(), rel=1e-9)
    # Every fragment sits cleanly in one floor group (no -1 with full coverage).
    assert set(np.unique(out.groupings["floor"].indices).tolist()) == {0, 1}


def test_face_cut_inherits_parent_fields():
    cp_per_tri = np.array([10.0, 20.0])
    ds = _wall(cp_per_tri)
    params = FaceCutParams(z_intervals=[0.0, 1.0, 2.0], name="floor")
    out = face_cut(ds, params)

    # Independently reconstruct the parent-per-fragment map from the same core.
    tri_verts = ds.topology.vertices[ds.topology.connectivity]
    _v, _n, parent_per_fragment = slice_triangles_with_parents(
        tri_verts,
        np.asarray(ds.elements.normal, dtype=np.float64),
        np.arange(ds.n_elements, dtype=np.int64),
        ([float("-inf"), float("inf")], [float("-inf"), float("inf")], [0.0, 1.0, 2.0]),
    )
    expected = ds.fields.read("cp")[parent_per_fragment]
    np.testing.assert_array_equal(out.fields.read("cp"), expected)


def test_face_cut_conserves_total_integrated_force():
    """Sum of per-fragment force equals the whole-wall force (exactness)."""
    ds = _wall(np.array([3.0, -2.0]))
    out = face_cut(ds, FaceCutParams(z_intervals=[0.0, 1.0, 2.0], name="floor"))

    fp = ForceContributionParams(nominal_area=1.0, directions=["y"])
    parent_cf = force_contribution(ds, fp).fields.read("cf_y")
    frag_cf = force_contribution(out, fp).fields.read("cf_y")

    # Total integrated force is invariant under slicing (area partitioned,
    # cp + normal inherited).
    np.testing.assert_allclose(frag_cf.sum(axis=0), parent_cf.sum(axis=0), rtol=1e-9)


def test_face_cut_is_exact_where_centroid_zoning_is_not():
    """A triangle straddling the boundary splits, unlike centroid zoning.

    The lower-left triangle [(0,0,0),(2,0,0),(2,0,2)] has centroid z = 2/3
    (floor 0), but ~part of its area lies above z = 1. face_cut assigns
    that upper part to floor 1; a centroid rule would put the whole
    triangle in floor 0.
    """
    ds = _wall(np.array([1.0, 1.0]))
    out = face_cut(ds, FaceCutParams(z_intervals=[0.0, 1.0, 2.0], name="floor"))

    idx = out.groupings["floor"].indices
    area_floor0 = out.elements.area[idx == 0].sum()
    area_floor1 = out.elements.area[idx == 1].sum()
    # Both floors receive real area; neither is the whole wall.
    total = _parent_area(ds).sum()
    assert 0.0 < area_floor0 < total
    assert 0.0 < area_floor1 < total
    assert area_floor0 + area_floor1 == pytest.approx(total, rel=1e-9)


def test_face_cut_unassigned_policy():
    # Cover only z in [0, 1): the top half of the wall falls outside.
    ds = _wall(np.array([1.0, 1.0]))
    dropped = face_cut(ds, FaceCutParams(z_intervals=[0.0, 1.0], unassigned_policy="drop"))
    kept = face_cut(
        ds, FaceCutParams(z_intervals=[0.0, 1.0], unassigned_policy="keep_as_unassigned")
    )

    assert (dropped.groupings["floor"].indices >= 0).all()
    assert (kept.groupings["floor"].indices == -1).any()
    # Dropping removes area; keeping retains the full wall.
    assert kept.elements.area.sum() == pytest.approx(_parent_area(ds).sum(), rel=1e-9)
    assert dropped.elements.area.sum() < kept.elements.area.sum()


def test_face_cut_requires_normal():
    ds = _wall(np.array([1.0, 1.0]))
    no_normal = ds.model_copy(update={"elements": ElementMeta(area=ds.elements.area)})
    with pytest.raises(ValueError, match="normal"):
        face_cut(no_normal, FaceCutParams(z_intervals=[0.0, 1.0, 2.0]))


def test_face_cut_requires_triangle_topology():
    ds = _wall(np.array([1.0, 1.0]))
    with pytest.raises(ValueError, match="triangle"):
        face_cut(
            ds.model_copy(update={"topology": None}),
            FaceCutParams(z_intervals=[0.0, 1.0, 2.0]),
        )


def test_face_cut_registered_in_catalog():
    from cfdmod.core.pipeline_yaml import op_info

    info = op_info("face_cut")
    assert info.produces == "surface"
    assert info.consumes == ["surface"]
    assert info.family == "source_create"
    assert "normal" in info.requires_element_meta
