import pathlib
import xml.etree.ElementTree as ET

import h5py
import numpy as np
import pytest

from cfdmod.io.xdmf import (
    filter_keys_by_range,
    get_pressure_keys,
    read_processing_metadata,
    read_step,
    read_timeseries_meta,
    write_processing_metadata,
    write_stats_field,
    write_stats_xdmf,
    write_temporal_xdmf,
    write_timeseries_geometry,
    write_timeseries_meta,
    write_timeseries_step,
)


@pytest.fixture()
def triangles():
    return np.array([[0, 1, 2], [1, 3, 2]], dtype=np.int32)


@pytest.fixture()
def vertices():
    return np.array(
        [[0, 0, 0], [0, 1, 0], [1, 0, 0], [1, 1, 0]], dtype=np.float64
    )


@pytest.fixture()
def timeseries_h5(tmp_path, triangles, vertices):
    path = tmp_path / "ts.h5"
    write_timeseries_geometry(path, triangles, vertices)
    write_timeseries_step(path, "pressure", "t0.0", np.array([1.0, 2.0]))
    write_timeseries_step(path, "pressure", "t0.5", np.array([1.5, 2.5]))
    write_timeseries_step(path, "pressure", "t1.0", np.array([2.0, 3.0]))
    write_timeseries_meta(
        path,
        time_steps=np.array([0.0, 0.5, 1.0]),
        time_normalized=np.array([0.0, 0.5, 1.0]),
    )
    return path


def test_get_pressure_keys_returns_sorted_pairs(timeseries_h5):
    keys = get_pressure_keys(timeseries_h5)
    assert keys == [(0.0, "t0.0"), (0.5, "t0.5"), (1.0, "t1.0")]


def test_get_pressure_keys_custom_group(tmp_path, triangles, vertices):
    path = tmp_path / "ts.h5"
    write_timeseries_geometry(path, triangles, vertices)
    write_timeseries_step(path, "cp", "t2.0", np.array([0.1, 0.2]))
    write_timeseries_step(path, "cp", "t0.5", np.array([0.3, 0.4]))
    keys = get_pressure_keys(path, group="cp")
    assert keys == [(0.5, "t0.5"), (2.0, "t2.0")]


def test_filter_keys_by_range_inclusive():
    keys = [(0.0, "t0.0"), (0.5, "t0.5"), (1.0, "t1.0"), (1.5, "t1.5")]
    assert filter_keys_by_range(keys, (0.5, 1.0)) == [(0.5, "t0.5"), (1.0, "t1.0")]
    assert filter_keys_by_range(keys, (0.0, 1.5)) == keys
    assert filter_keys_by_range(keys, (10.0, 20.0)) == []


def test_read_step_returns_array(timeseries_h5):
    arr = read_step(timeseries_h5, "t0.5", group="pressure")
    np.testing.assert_array_equal(arr, np.array([1.5, 2.5]))


def test_read_timeseries_meta_basic(timeseries_h5):
    meta = read_timeseries_meta(timeseries_h5)
    np.testing.assert_array_equal(meta["time_steps"], [0.0, 0.5, 1.0])
    np.testing.assert_array_equal(meta["time_normalized"], [0.0, 0.5, 1.0])
    assert "region_labels" not in meta


def test_read_timeseries_meta_with_region_labels(tmp_path):
    path = tmp_path / "ts.h5"
    write_timeseries_meta(
        path,
        time_steps=np.array([0.0, 1.0]),
        time_normalized=np.array([0.0, 1.0]),
        region_labels=["roof", "wall"],
    )
    meta = read_timeseries_meta(path)
    assert meta["region_labels"] == ["roof", "wall"]


def test_write_timeseries_step_overwrites_existing_key(tmp_path):
    path = tmp_path / "ts.h5"
    write_timeseries_step(path, "pressure", "t0.0", np.array([1.0, 2.0]))
    write_timeseries_step(path, "pressure", "t0.0", np.array([9.0, 9.0]))
    np.testing.assert_array_equal(
        read_step(path, "t0.0", group="pressure"), [9.0, 9.0]
    )


def test_write_timeseries_meta_overwrites_existing(tmp_path):
    path = tmp_path / "ts.h5"
    write_timeseries_meta(
        path, time_steps=np.array([0.0]), time_normalized=np.array([0.0])
    )
    write_timeseries_meta(
        path,
        time_steps=np.array([0.0, 1.0]),
        time_normalized=np.array([0.0, 0.5]),
        region_labels=["a"],
    )
    meta = read_timeseries_meta(path)
    np.testing.assert_array_equal(meta["time_steps"], [0.0, 1.0])
    assert meta["region_labels"] == ["a"]


def test_write_timeseries_geometry_overwrites(tmp_path, triangles, vertices):
    path = tmp_path / "ts.h5"
    write_timeseries_geometry(path, triangles, vertices)
    new_tri = np.array([[0, 1, 2]], dtype=np.int32)
    new_verts = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0]], dtype=np.float64)
    write_timeseries_geometry(path, new_tri, new_verts)
    with h5py.File(path, "r") as f:
        assert f["Triangles"].shape == (1, 3)
        assert f["Geometry"].shape == (3, 3)


def test_write_temporal_xdmf_single_group(tmp_path, timeseries_h5):
    xdmf_path = tmp_path / "ts.xdmf"
    write_temporal_xdmf(timeseries_h5, xdmf_path, group="pressure")

    tree = ET.parse(xdmf_path)
    root = tree.getroot()
    assert root.tag == "Xdmf"
    grids = root.findall(".//Grid[@GridType='Uniform']")
    assert len(grids) == 3

    times = [float(g.find("Time").attrib["Value"]) for g in grids]
    assert times == sorted(times)
    assert times == [0.0, 0.5, 1.0]

    h5_name = timeseries_h5.name
    attrs = grids[0].findall("Attribute")
    assert len(attrs) == 1
    assert attrs[0].attrib["Name"] == "pressure"
    assert attrs[0].find("DataItem").text == f"{h5_name}:/pressure/t0.0"


def test_write_temporal_xdmf_multi_group(tmp_path, triangles, vertices):
    h5 = tmp_path / "ts.h5"
    write_timeseries_geometry(h5, triangles, vertices)
    for direction in ("cf_x", "cf_y", "cf_z"):
        write_timeseries_step(h5, direction, "t0.0", np.array([1.0, 2.0]))
        write_timeseries_step(h5, direction, "t1.0", np.array([3.0, 4.0]))
    write_timeseries_meta(
        h5, time_steps=np.array([0.0, 1.0]), time_normalized=np.array([0.0, 1.0])
    )

    xdmf_path = tmp_path / "ts.xdmf"
    write_temporal_xdmf(h5, xdmf_path, group=["cf_x", "cf_y", "cf_z"])

    grids = ET.parse(xdmf_path).getroot().findall(".//Grid[@GridType='Uniform']")
    assert len(grids) == 2
    attr_names = [a.attrib["Name"] for a in grids[0].findall("Attribute")]
    assert attr_names == ["cf_x", "cf_y", "cf_z"]
    refs = [a.find("DataItem").text for a in grids[0].findall("Attribute")]
    assert refs == [f"{h5.name}:/cf_x/t0.0", f"{h5.name}:/cf_y/t0.0", f"{h5.name}:/cf_z/t0.0"]


def test_write_stats_field_embeds_group_mesh(tmp_path, triangles, vertices):
    path = tmp_path / "results.h5"
    write_stats_field(
        path,
        group="cp/case1",
        stat_name="mean",
        values=np.array([0.1, 0.2]),
        triangles=triangles,
        vertices=vertices,
    )
    with h5py.File(path, "r") as f:
        assert "Triangles" not in f
        assert "Geometry" not in f
        assert f["cp/case1/Triangles"].shape == triangles.shape
        assert f["cp/case1/Geometry"].shape == vertices.shape
        np.testing.assert_array_equal(f["cp/case1/mean"][:], [0.1, 0.2])


def test_write_stats_field_does_not_overwrite_group_mesh(tmp_path, triangles, vertices):
    path = tmp_path / "results.h5"
    write_stats_field(
        path,
        group="cp/case1",
        stat_name="mean",
        values=np.array([0.1, 0.2]),
        triangles=triangles,
        vertices=vertices,
    )
    write_stats_field(
        path,
        group="cp/case1",
        stat_name="std",
        values=np.array([0.01, 0.02]),
        triangles=np.zeros((9, 3), dtype=np.int32),
        vertices=np.zeros((9, 3), dtype=np.float64),
    )
    with h5py.File(path, "r") as f:
        assert f["cp/case1/Triangles"].shape == triangles.shape
        np.testing.assert_array_equal(f["cp/case1/std"][:], [0.01, 0.02])


def test_write_stats_field_overwrites_existing_stat(tmp_path, triangles, vertices):
    path = tmp_path / "results.h5"
    write_stats_field(
        path,
        group="cp",
        stat_name="mean",
        values=np.array([1.0, 1.0]),
        triangles=triangles,
        vertices=vertices,
    )
    write_stats_field(
        path,
        group="cp",
        stat_name="mean",
        values=np.array([9.0, 9.0]),
    )
    with h5py.File(path, "r") as f:
        np.testing.assert_array_equal(f["cp/mean"][:], [9.0, 9.0])


def test_write_stats_xdmf_emits_one_grid_per_meshed_group(tmp_path, triangles, vertices):
    path = tmp_path / "results.h5"
    cp_tri, cp_verts = triangles, vertices
    body_tri = np.array([[0, 1, 2]], dtype=np.int32)
    body_verts = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0]], dtype=np.float64)

    write_stats_field(
        path, "cp/default", "mean", np.array([0.1, 0.2]),
        triangles=cp_tri, vertices=cp_verts,
    )
    write_stats_field(path, "cp/default", "rms", np.array([0.01, 0.02]))
    write_stats_field(
        path, "cf_x/m1/body", "mean", np.array([0.5]),
        triangles=body_tri, vertices=body_verts,
    )

    xdmf_path = tmp_path / "results.xdmf"
    write_stats_xdmf(path, xdmf_path)

    root = ET.parse(xdmf_path).getroot()
    grids = root.findall("Domain/Grid")
    grid_names = sorted(g.attrib["Name"] for g in grids)
    assert grid_names == ["cf_x/m1/body", "cp/default"]

    cp_grid = next(g for g in grids if g.attrib["Name"] == "cp/default")
    cp_attrs = sorted(a.attrib["Name"] for a in cp_grid.findall("Attribute"))
    assert cp_attrs == ["cp/default/mean", "cp/default/rms"]
    cp_topo = cp_grid.find("Topology/DataItem")
    assert cp_topo.text == f"{path.name}:/cp/default/Triangles"

    body_grid = next(g for g in grids if g.attrib["Name"] == "cf_x/m1/body")
    body_topo = body_grid.find("Topology")
    assert int(body_topo.attrib["NumberOfElements"]) == 1


def test_write_stats_xdmf_skips_groups_without_geometry(tmp_path, triangles, vertices):
    path = tmp_path / "results.h5"
    write_stats_field(
        path, "cp/default", "mean", np.array([0.1, 0.2]),
        triangles=triangles, vertices=vertices,
    )
    # Add a meta-only group: should NOT produce a grid.
    write_timeseries_meta(
        path, time_steps=np.array([0.0]), time_normalized=np.array([0.0])
    )

    xdmf_path = tmp_path / "results.xdmf"
    write_stats_xdmf(path, xdmf_path)
    grid_names = [g.attrib["Name"] for g in ET.parse(xdmf_path).getroot().findall("Domain/Grid")]
    assert grid_names == ["cp/default"]


def test_write_stats_xdmf_raises_when_no_grids(tmp_path):
    path = tmp_path / "results.h5"
    write_timeseries_meta(
        path, time_steps=np.array([0.0]), time_normalized=np.array([0.0])
    )
    with pytest.raises(ValueError, match="no group with both Triangles and Geometry"):
        write_stats_xdmf(path, tmp_path / "results.xdmf")


def test_xdmf_files_have_doctype_header(tmp_path, timeseries_h5):
    xdmf_path = tmp_path / "ts.xdmf"
    write_temporal_xdmf(timeseries_h5, xdmf_path, group="pressure")
    text = pathlib.Path(xdmf_path).read_text()
    assert text.startswith('<?xml version="1.0" ?>\n<!DOCTYPE Xdmf SYSTEM "Xdmf.dtd">')


def test_processing_metadata_round_trip(tmp_path):
    path = tmp_path / "out.h5"
    cfg = {
        "statistics": [{"stats": "mean"}, {"stats": "rms"}],
        "simul_U_H": 1.5,
        "macroscopic_type": "rho",
    }
    write_processing_metadata(
        path,
        "cp/default",
        cfg,
        extra={"coefficient": "cp", "cfg_lbl": "default", "body_h5": "/tmp/body.h5"},
    )
    md = read_processing_metadata(path, "cp/default")
    assert md["config"] == cfg
    assert md["coefficient"] == "cp"
    assert md["cfg_lbl"] == "default"
    assert md["body_h5"] == "/tmp/body.h5"
    assert "produced_at" in md
    assert "cfdmod_version" in md


def test_processing_metadata_does_not_break_stats_xdmf(tmp_path, triangles, vertices):
    path = tmp_path / "results.h5"
    write_stats_field(
        path,
        group="cp/default",
        stat_name="mean",
        values=np.array([0.1, 0.2]),
        triangles=triangles,
        vertices=vertices,
    )
    write_processing_metadata(path, "cp/default", {"simul_U_H": 1.0})

    xdmf_path = tmp_path / "results.xdmf"
    write_stats_xdmf(path, xdmf_path)

    root = ET.parse(xdmf_path).getroot()
    grids = root.findall("Domain/Grid")
    assert [g.attrib["Name"] for g in grids] == ["cp/default"]
    attrs = [a.attrib["Name"] for a in grids[0].findall("Attribute")]
    assert attrs == ["cp/default/mean"]


def test_processing_metadata_overwrites_existing_yaml(tmp_path):
    path = tmp_path / "out.h5"
    write_processing_metadata(path, "cp", {"x": 1})
    write_processing_metadata(path, "cp", {"x": 2})
    md = read_processing_metadata(path, "cp")
    assert md["config"] == {"x": 2}
