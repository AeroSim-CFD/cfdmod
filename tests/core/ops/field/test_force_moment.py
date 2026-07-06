"""Known-answer tests for the force / moment physics ops.

These pin the actual aerodynamic math (sign, projection, lever arm,
nominal-area/volume normalisation) against hand-computed values, on a
tiny two-triangle source. Previously only shapes were asserted, so a
sign flip or a wrong lever arm would have gone unnoticed.
"""

from __future__ import annotations

import pathlib

import numpy as np
from lnas import LnasFormat, LnasGeometry

from cfdmod.adapters.memory import MemoryFieldStore
from cfdmod.core import ElementMeta, SurfaceDataSource, TimeAxis, Topology
from cfdmod.core.ops.field import (
    ForceContributionParams,
    MomentContributionParams,
    force_contribution,
    moment_contribution,
)
from cfdmod.core.ops.geometric import MeshAttachParams, mesh_attach


def _two_tri_source(cp: np.ndarray, area, normal, position) -> SurfaceDataSource:
    """Two-triangle surface with caller-supplied element metadata.

    The topology vertices are dummy (only the count matters); the areas
    / normals / centroids that drive the physics come from ``elements``.
    """
    verts = np.array(
        [[0, 0, 0], [1, 0, 0], [1, 1, 0], [0, 1, 0]],
        dtype=np.float64,
    )
    tris = np.array([[0, 1, 2], [0, 2, 3]], dtype=np.int32)
    return SurfaceDataSource(
        time=TimeAxis(initial_time=0.0, timestep_size=1.0, n_timesteps=cp.shape[1]),
        topology=Topology.triangles(tris, verts),
        elements=ElementMeta(
            area=np.asarray(area, dtype=np.float64),
            normal=np.asarray(normal, dtype=np.float64),
            position=np.asarray(position, dtype=np.float64),
        ),
        fields=MemoryFieldStore({"cp": np.asarray(cp, dtype=np.float64)}),
    )


def test_force_contribution_known_answer():
    # cf_dir = -cp * area * normal_dir / nominal_area
    cp = np.array([[0.5], [2.0]])
    area = [2.0, 3.0]
    normal = [[1.0, 0.0, 0.0], [0.0, 0.0, 1.0]]
    position = [[1.0, 0.0, 0.0], [0.0, 2.0, 4.0]]
    ds = _two_tri_source(cp, area, normal, position)

    out = force_contribution(ds, ForceContributionParams(field="cp", nominal_area=10.0))

    # tri0: -0.5*2*1/10 = -0.1 ; tri1: -2*3*0/10 = 0
    assert np.allclose(out.fields.read("cf_x"), [[-0.1], [0.0]])
    # no triangle has a y-normal component
    assert np.allclose(out.fields.read("cf_y"), [[0.0], [0.0]])
    # tri0: 0 ; tri1: -2*3*1/10 = -0.6
    assert np.allclose(out.fields.read("cf_z"), [[0.0], [-0.6]])


def test_moment_contribution_known_answer():
    cp = np.array([[0.5], [2.0]])
    area = [2.0, 3.0]
    normal = [[1.0, 0.0, 0.0], [0.0, 0.0, 1.0]]
    position = [[1.0, 0.0, 0.0], [0.0, 2.0, 4.0]]
    ds = _two_tri_source(cp, area, normal, position)
    ds = force_contribution(ds, ForceContributionParams(field="cp", nominal_area=10.0))

    out = moment_contribution(
        ds,
        MomentContributionParams(
            lever_origin=(0.0, 0.0, 0.0),
            nominal_area=10.0,
            nominal_volume=5.0,
        ),
    )

    # Per-element force (Cf * nominal_area): tri0 f=(-1,0,0), tri1 f=(0,0,-6).
    # m = r x f, /nominal_volume.
    #   tri0: r=(1,0,0) x (-1,0,0) = 0
    #   tri1: r=(0,2,4) x (0,0,-6) = (2*-6 - 4*0, 4*0 - 0*-6, 0*0 - 2*0)
    #        = (-12, 0, 0); /5 -> cm_x = -2.4
    assert np.allclose(out.fields.read("cm_x"), [[0.0], [-2.4]])
    assert np.allclose(out.fields.read("cm_y"), [[0.0], [0.0]])
    assert np.allclose(out.fields.read("cm_z"), [[0.0], [0.0]])


def test_mesh_attach_areas_normals_centroids(tmp_path: pathlib.Path):
    # One triangle in the z=0 plane: (0,0,0), (1,0,0), (0,1,0).
    # area = 0.5, unit normal = +z, centroid = (1/3, 1/3, 0).
    verts = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0]], dtype=np.float32)
    tris = np.array([[0, 1, 2]], dtype=np.uint32)
    lnas = LnasFormat(
        version="v0.5.0",
        geometry=LnasGeometry(vertices=verts, triangles=tris),
        surfaces={"s": np.array([0], dtype=np.uint32)},
    )
    mesh_path = tmp_path / "one_tri.lnas"
    lnas.to_file(mesh_path)

    ds = SurfaceDataSource(
        time=TimeAxis(initial_time=0.0, timestep_size=1.0, n_timesteps=1),
        topology=Topology.triangles(tris.astype(np.int32), verts.astype(np.float64)),
        elements=ElementMeta(),
        fields=MemoryFieldStore({"cp": np.zeros((1, 1))}),
    )
    out = mesh_attach(ds, MeshAttachParams(mesh=str(mesh_path)))

    assert np.allclose(out.elements.area, [0.5])
    assert np.allclose(np.abs(out.elements.normal[0]), [0.0, 0.0, 1.0])
    assert np.allclose(out.elements.position, [[1 / 3, 1 / 3, 0.0]])
