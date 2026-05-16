"""Remesh-module tests on the real container-pack fixture.

The synthetic tests in ``test_functions.py`` cover the algorithm's edge cases
(coplanar square, curved patch, L-shape, two-component bridge). These tests
exercise the same operations on the actual sliced-then-per-face mesh
extracted from a wind-tunnel container pack -- i.e., the input ``remesh_per_group``
is normally called on in the production notebook.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pytest
from lnas import LnasFormat

from cfdmod import (
    ByConnectivityGrouping,
    BySizeRoundedPerComponent,
    RegroupConfig,
    mesh_from_h5,
)
from cfdmod.geometry.grouping import GroupingResult
from cfdmod.regroup import build_regroup_mapping, expand_regroup_chain
from cfdmod.regroup.functions import build_sliced_regrouped_mesh
from cfdmod.remesh import merge_coplanar, remesh_per_group
from tests.remesh.conftest import FIXTURE_BODY


@dataclass
class _PerFaceBundle:
    per_face_lnas: LnasFormat
    n_parents: int
    n_fragments: int


@pytest.fixture(scope="module")
def per_face_lnas() -> _PerFaceBundle:
    """Sliced + per-face-bucketed LnasFormat for the fixture body H5.

    This is the exact intermediate the notebook pipeline feeds into
    ``remesh_per_group``. Module-scoped so the (already-cheap) slicing
    runs once for the whole test file.
    """
    mesh = mesh_from_h5(FIXTURE_BODY)
    cfg = RegroupConfig(
        groupings=[
            ByConnectivityGrouping(name_template="cc{idx}", min_triangles=4),
            BySizeRoundedPerComponent(
                target_size_x=6.34,
                target_size_y=2.58,
                target_size_z=2.6,
                name_template="{parent}_c{idx}",
            ),
        ],
        aggregation="sliced",
        timeseries_group="cp",
        output_geometry_format="lnas",
        unassigned_policy="drop",
    )
    expanded, consumed, parent_intervals, parent_triangles = expand_regroup_chain(
        cfg.groupings, mesh, cfg.transformation
    )
    grouping = build_regroup_mapping(mesh, expanded, cfg.transformation)
    if consumed:
        grouping = GroupingResult(
            parent_n_triangles=grouping.parent_n_triangles,
            groups={n: i for n, i in grouping.groups.items() if n not in consumed},
        )
    sliced, _ = build_sliced_regrouped_mesh(
        mesh,
        grouping,
        parent_intervals=parent_intervals,
        parent_triangles=parent_triangles,
        unassigned_policy="drop",
    )

    n_frag = sliced.geometry.triangles.shape[0]
    tri_v = sliced.geometry.triangle_vertices
    crosses = np.cross(tri_v[:, 1] - tri_v[:, 0], tri_v[:, 2] - tri_v[:, 0])
    norms = crosses / (np.linalg.norm(crosses, axis=1, keepdims=True) + 1e-30)
    axis_idx = np.abs(norms).argmax(axis=1)
    sign_neg = norms[np.arange(n_frag), axis_idx] < 0
    direction = (axis_idx * 2 + sign_neg.astype(np.int64)).astype(np.int64)

    cell_names = sorted(sliced.surfaces.keys())
    cell_of = np.empty(n_frag, dtype=np.int64)
    for ci, name in enumerate(cell_names):
        cell_of[sliced.surfaces[name]] = ci

    group_key = cell_of * 6 + direction
    unique_groups, bucket_of = np.unique(group_key, return_inverse=True)
    direction_suffix = {0: "xp", 1: "xn", 2: "yp", 3: "yn", 4: "zp", 5: "zn"}
    per_face_surfaces: dict[str, np.ndarray] = {}
    for gi, gkey in enumerate(unique_groups):
        per_face_surfaces[f"{cell_names[gkey // 6]}_{direction_suffix[gkey % 6]}"] = (
            np.flatnonzero(bucket_of == gi).astype(np.int32)
        )
    lnas = LnasFormat(version=sliced.version, geometry=sliced.geometry, surfaces=per_face_surfaces)
    return _PerFaceBundle(
        per_face_lnas=lnas, n_parents=mesh.geometry.triangles.shape[0], n_fragments=n_frag
    )


def _surface_area(lnas: LnasFormat, surface_name: str) -> float:
    idxs = lnas.surfaces[surface_name]
    if idxs.size == 0:
        return 0.0
    tv = lnas.geometry.triangle_vertices[idxs]
    crosses = np.cross(tv[:, 1] - tv[:, 0], tv[:, 2] - tv[:, 0])
    return float(np.linalg.norm(crosses, axis=1).sum() / 2)


@pytest.mark.integration
def test_fixture_per_face_intermediate_shape(per_face_lnas):
    """Pin the intermediate cardinalities so changes to the slicer/regroup
    don't silently shift them out from under the remesh tests below."""
    bundle = per_face_lnas
    assert bundle.n_parents == 121  # 6 selected containers' triangles
    assert bundle.n_fragments == 248  # post-slicing fan-out
    assert len(bundle.per_face_lnas.surfaces) == 13  # (cell, face) buckets


@pytest.mark.integration
def test_merge_coplanar_on_one_fixture_face_is_lossless(per_face_lnas):
    """Pick the surface with the most fragments, apply ``merge_coplanar`` directly,
    and assert it collapses to far fewer triangles while preserving area exactly."""
    lnas = per_face_lnas.per_face_lnas
    biggest_name = max(lnas.surfaces, key=lambda n: lnas.surfaces[n].size)
    idxs = lnas.surfaces[biggest_name]

    sub_tris_parent = lnas.geometry.triangles[idxs]
    used = np.unique(sub_tris_parent)
    remap = np.full(lnas.geometry.vertices.shape[0], -1, dtype=np.int64)
    remap[used] = np.arange(used.size, dtype=np.int64)
    sub_verts = lnas.geometry.vertices[used].astype(np.float64)
    sub_tris = remap[sub_tris_parent].astype(np.int32)

    area_in = float(
        np.linalg.norm(
            np.cross(
                sub_verts[sub_tris[:, 1]] - sub_verts[sub_tris[:, 0]],
                sub_verts[sub_tris[:, 2]] - sub_verts[sub_tris[:, 0]],
            ),
            axis=1,
        ).sum()
        / 2
    )

    new_v, new_t = merge_coplanar(sub_verts, sub_tris)
    area_out = float(
        np.linalg.norm(
            np.cross(
                new_v[new_t[:, 1]] - new_v[new_t[:, 0]],
                new_v[new_t[:, 2]] - new_v[new_t[:, 0]],
            ),
            axis=1,
        ).sum()
        / 2
    )

    assert new_t.shape[0] < sub_tris.shape[0]
    # Coplanar merge is lossless -- exact area preservation up to float epsilon.
    assert area_out == pytest.approx(area_in, rel=1e-9, abs=1e-12)


@pytest.mark.integration
def test_remesh_per_group_preserves_area_per_surface(per_face_lnas):
    """Per-surface area is unchanged through ``remesh_per_group`` defaults."""
    lnas = per_face_lnas.per_face_lnas
    remeshed = remesh_per_group(lnas)
    max_drift = 0.0
    for name in lnas.surfaces:
        a_in = _surface_area(lnas, name)
        a_out = _surface_area(remeshed, name)
        if a_in > 0:
            max_drift = max(max_drift, abs(a_out - a_in) / a_in)
    assert max_drift < 1e-9


@pytest.mark.integration
def test_remesh_per_group_preserves_surface_names(per_face_lnas):
    """Surface name set is preserved one-to-one (zero-area buckets may stay as empty entries)."""
    lnas = per_face_lnas.per_face_lnas
    remeshed = remesh_per_group(lnas)
    in_names = set(lnas.surfaces.keys())
    out_names = set(remeshed.surfaces.keys())
    assert in_names == out_names


@pytest.mark.integration
def test_remesh_per_group_reduces_triangle_count_substantially(per_face_lnas):
    """The fixture's per-face mesh has 248 fragments; coplanar merge collapses them
    to ~60 triangles across the 13 surfaces -- a > 70% reduction.
    """
    lnas = per_face_lnas.per_face_lnas
    n_in = lnas.geometry.triangles.shape[0]
    remeshed = remesh_per_group(lnas)
    n_out = remeshed.geometry.triangles.shape[0]
    assert n_out < n_in
    reduction = 1.0 - n_out / n_in
    assert reduction > 0.7
    # Sanity: every surface should have at least 1 surviving triangle, and the
    # minimum-triangulation lower bound (n_boundary_vertices - 2) means we never
    # go below 2 per non-empty surface in practice for these axis-aligned faces.
    nonempty = [s for s in remeshed.surfaces.values() if s.size > 0]
    assert all(s.size >= 2 for s in nonempty)


@pytest.mark.integration
def test_remesh_per_group_default_is_idempotent_on_fixture(per_face_lnas):
    """Running coplanar merge twice does not collapse further: the first pass
    already reaches the minimum triangulation of each coplanar component, so
    re-running it is a no-op on the already-coarse mesh.
    """
    lnas = per_face_lnas.per_face_lnas
    once = remesh_per_group(lnas)
    twice = remesh_per_group(once)
    assert twice.geometry.triangles.shape[0] == once.geometry.triangles.shape[0]
    assert set(twice.surfaces.keys()) == set(once.surfaces.keys())
