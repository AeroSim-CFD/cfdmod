"""Unit tests for the filter_by_submesh op."""

from __future__ import annotations

import pathlib

import numpy as np
import pytest

from cfdmod.adapters.memory import MemoryFieldStore
from cfdmod.core import (
    ElementMeta,
    Grouping,
    SurfaceDataSource,
    TimeAxis,
    Topology,
)
from cfdmod.core.ops.data_source_create import FilterBySubmeshParams, filter_by_submesh
from cfdmod.io.geometry import export_stl

GALPAO = pathlib.Path("fixtures/tests/pressure/galpao")


def _strip_surface(n: int = 6) -> SurfaceDataSource:
    """``n`` unit triangles along x, each with its own pressure history."""
    verts = np.concatenate(
        [
            np.array([[2.0 * i, 0.0, 0.0], [2.0 * i + 1.0, 0.0, 0.0], [2.0 * i, 1.0, 0.0]])
            for i in range(n)
        ]
    )
    tris = np.arange(3 * n, dtype=np.int32).reshape(n, 3)
    pressure = np.arange(n * 3, dtype=np.float64).reshape(n, 3)
    return SurfaceDataSource(
        time=TimeAxis(initial_time=0.0, timestep_size=1.0, n_timesteps=3),
        topology=Topology.triangles(tris, verts),
        elements=ElementMeta(area=np.arange(n, dtype=np.float64)),
        groupings={"body": Grouping(name="body", indices=list(range(n)))},
        fields=MemoryFieldStore({"pressure": pressure}),
    )


def _write_sub_stl(ds: SurfaceDataSource, picked: list[int], path: pathlib.Path) -> pathlib.Path:
    """Export the picked reference triangles as a standalone (float32) STL."""
    tris = ds.topology.vertices[ds.topology.connectivity][picked]
    normals = np.zeros((len(picked), 3), dtype=np.float32)
    export_stl(path, tris.astype(np.float32), normals)
    return path


def test_filter_keeps_the_submesh_triangles_in_submesh_order(tmp_path):
    ds = _strip_surface()
    picked = [4, 1, 5]
    mesh = _write_sub_stl(ds, picked, tmp_path / "region.stl")

    out = filter_by_submesh(ds, FilterBySubmeshParams(mesh=str(mesh)))

    assert out.n_elements == 3
    np.testing.assert_array_equal(out.fields.read("pressure"), ds.fields.read("pressure")[picked])
    np.testing.assert_array_equal(out.elements.area, picked)
    np.testing.assert_array_equal(out.groupings["body"].indices, picked)
    np.testing.assert_array_equal(out.topology.connectivity, ds.topology.connectivity[picked])


def test_explicit_reference_mesh_matches_the_topology_route(tmp_path):
    ds = _strip_surface()
    ref = _write_sub_stl(ds, list(range(6)), tmp_path / "reference.stl")
    mesh = _write_sub_stl(ds, [3, 0], tmp_path / "region.stl")

    from_path = filter_by_submesh(
        ds, FilterBySubmeshParams(mesh=str(mesh), reference_mesh=str(ref))
    )
    from_topology = filter_by_submesh(ds, FilterBySubmeshParams(mesh=str(mesh)))
    np.testing.assert_array_equal(
        from_path.fields.read("pressure"), from_topology.fields.read("pressure")
    )
    np.testing.assert_array_equal(from_path.elements.area, [3.0, 0.0])


def test_unmatched_triangle_raises_by_default(tmp_path):
    ds = _strip_surface()
    stranger = ds.topology.vertices[ds.topology.connectivity][[2]] + 500.0
    tris = np.concatenate([ds.topology.vertices[ds.topology.connectivity][[1]], stranger])
    mesh = tmp_path / "region.stl"
    export_stl(mesh, tris.astype(np.float32), np.zeros((2, 3), dtype=np.float32))

    with pytest.raises(ValueError, match="no match in the reference geometry"):
        filter_by_submesh(ds, FilterBySubmeshParams(mesh=str(mesh)))

    out = filter_by_submesh(ds, FilterBySubmeshParams(mesh=str(mesh), on_missing="drop"))
    assert out.n_elements == 1
    np.testing.assert_array_equal(out.elements.area, [1.0])


def test_nothing_matched_raises(tmp_path):
    ds = _strip_surface()
    tris = ds.topology.vertices[ds.topology.connectivity][[0]] + 500.0
    mesh = tmp_path / "elsewhere.stl"
    export_stl(mesh, tris.astype(np.float32), np.zeros((1, 3), dtype=np.float32))

    with pytest.raises(ValueError, match="selects 0 elements"):
        filter_by_submesh(ds, FilterBySubmeshParams(mesh=str(mesh), on_missing="drop"))


def test_reference_mesh_with_the_wrong_element_count_raises(tmp_path):
    ds = _strip_surface()
    ref = _write_sub_stl(ds, [0, 1, 2], tmp_path / "half_reference.stl")
    mesh = _write_sub_stl(ds, [1], tmp_path / "region.stl")

    with pytest.raises(ValueError, match="but data source has 6 elements"):
        filter_by_submesh(ds, FilterBySubmeshParams(mesh=str(mesh), reference_mesh=str(ref)))


def test_op_is_registered_in_the_yaml_registry():
    from cfdmod.core.pipeline_yaml import OP_REGISTRY

    assert "filter_by_submesh" in OP_REGISTRY


@pytest.mark.integration
def test_galpao_region_stl_filters_the_reference_data_source():
    from lnas import LnasFormat

    from cfdmod.core.ops.geometric import MeshAttachParams, mesh_attach

    lnas = LnasFormat.from_file(GALPAO / "galpao.lnas")
    n = lnas.geometry.triangles.shape[0]
    ds = SurfaceDataSource(
        time=TimeAxis(initial_time=0.0, timestep_size=0.1, n_timesteps=2),
        topology=Topology.triangles(lnas.geometry.triangles, lnas.geometry.vertices),
        elements=ElementMeta(),
        fields=MemoryFieldStore({"cp": np.tile(np.arange(n, dtype=np.float64), (2, 1)).T}),
    )
    ds = mesh_attach(ds, MeshAttachParams(mesh=str(GALPAO / "galpao.lnas")))

    out = filter_by_submesh(ds, FilterBySubmeshParams(mesh=str(GALPAO / "L2_yp.stl")))

    assert out.n_elements == 256
    # The synthetic cp field is the reference triangle index, so the filtered
    # field is exactly the list of matched reference indices.
    matched = out.fields.read("cp")[:, 0].astype(np.int64)
    assert matched.min() >= 0 and matched.max() < n
    assert np.unique(matched).size == 256
    np.testing.assert_allclose(
        out.elements.area, np.asarray(lnas.geometry.areas)[matched], rtol=1e-12
    )
