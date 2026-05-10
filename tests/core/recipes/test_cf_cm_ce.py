"""Unit tests for the Cf, Cm, Ce recipes -- the per-group aggregation layer."""

from __future__ import annotations

import numpy as np

from cfdmod.adapters.memory import MemoryFieldStore
from cfdmod.core import (
    ElementMeta,
    Grouping,
    SurfaceDataSource,
    TimeAxis,
    Topology,
)
from cfdmod.core.recipes import (
    CeRecipeConfig,
    CfRecipeConfig,
    CmRecipeConfig,
    ce_pipeline,
    cf_pipeline,
    cm_pipeline,
)


def _two_body_surface() -> SurfaceDataSource:
    verts = np.array(
        [[0, 0, 0], [1, 0, 0], [0, 1, 0], [1, 1, 0], [0.5, 0.5, 1]], dtype=np.float64
    )
    tris = np.array([[0, 1, 2], [1, 3, 2], [2, 3, 4], [0, 2, 4]], dtype=np.int32)
    cp_x = np.array([[1.0], [2.0], [10.0], [20.0]])
    cp_y = np.array([[3.0], [4.0], [30.0], [40.0]])
    return SurfaceDataSource(
        time=TimeAxis(initial_time=0.0, timestep_size=1.0, n_timesteps=1),
        topology=Topology.triangles(tris, verts),
        elements=ElementMeta(area=np.array([1.0, 1.0, 2.0, 2.0])),
        groupings={"body": Grouping(name="body", indices=[0, 0, 1, 1])},
        fields=MemoryFieldStore({"cp_x": cp_x, "cp_y": cp_y}),
    )


def test_cf_pipeline_yields_per_body_per_direction_field():
    ds = _two_body_surface()
    out = cf_pipeline(CfRecipeConfig(grouping="body", directions=["x", "y"]))(ds)
    assert out.kind == "groups"
    assert out.n_elements == 2  # two bodies
    np.testing.assert_allclose(out.fields.read("cf_x")[:, 0], [1.5, 15.0])
    np.testing.assert_allclose(out.fields.read("cf_y")[:, 0], [3.5, 35.0])


def test_cm_pipeline_sums_within_each_group():
    ds = _two_body_surface()
    # Repurpose cp_* fields as moment contributions.
    ds = ds.with_field("cm_x", ds.fields.read("cp_x"))
    ds = ds.with_field("cm_y", ds.fields.read("cp_y"))
    out = cm_pipeline(CmRecipeConfig(grouping="body", directions=["x", "y"]))(ds)
    np.testing.assert_allclose(out.fields.read("cm_x")[:, 0], [3.0, 30.0])
    np.testing.assert_allclose(out.fields.read("cm_y")[:, 0], [7.0, 70.0])


def test_ce_pipeline_area_weighted_mean_per_zone():
    verts = np.array(
        [[0, 0, 0], [1, 0, 0], [0, 1, 0], [1, 1, 0]], dtype=np.float64
    )
    tris = np.array([[0, 1, 2], [1, 3, 2]], dtype=np.int32)
    cp = np.array([[2.0, 4.0], [4.0, 8.0]])
    ds = SurfaceDataSource(
        time=TimeAxis(initial_time=0.0, timestep_size=1.0, n_timesteps=2),
        topology=Topology.triangles(tris, verts),
        elements=ElementMeta(area=np.array([1.0, 3.0])),
        groupings={"zone": Grouping(name="zone", indices=[0, 0])},
        fields=MemoryFieldStore({"cp": cp}),
    )
    out = ce_pipeline(CeRecipeConfig(grouping="zone"))(ds)
    # area-weighted mean: (1*[2,4] + 3*[4,8]) / 4 = ([14,28]) / 4 = [3.5, 7.0]
    np.testing.assert_allclose(out.fields.read("ce")[0], [3.5, 7.0])
