import pathlib
import xml.etree.ElementTree as ET

import h5py
import numpy as np
import pytest

from cfdmod.io.xdmf import (
    filter_keys_by_range,
    get_pressure_keys,
    read_step,
    read_timeseries_meta,
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


def test_write_temporal_xdmf_structure(tmp_path, timeseries_h5):
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
    attr_item = grids[0].find("Attribute/DataItem")
    assert attr_item.text == f"{h5_name}:/pressure/t0.0"


def test_write_stats_field_creates_geometry_once(tmp_path, triangles, vertices):
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
        assert f["Triangles"].shape == triangles.shape
        np.testing.assert_array_equal(f["cp/case1/mean"][:], [0.1, 0.2])
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


def test_write_stats_xdmf_lists_all_fields(tmp_path, triangles, vertices):
    path = tmp_path / "results.h5"
    write_stats_field(
        path,
        group="cp",
        stat_name="mean",
        values=np.array([0.1, 0.2]),
        triangles=triangles,
        vertices=vertices,
    )
    write_stats_field(path, group="cp", stat_name="std", values=np.array([0.01, 0.02]))
    write_stats_field(path, group="ce", stat_name="mean", values=np.array([1.0, 2.0]))

    xdmf_path = tmp_path / "results.xdmf"
    write_stats_xdmf(path, xdmf_path)

    tree = ET.parse(xdmf_path)
    root = tree.getroot()
    attr_names = sorted(a.attrib["Name"] for a in root.findall(".//Attribute"))
    assert attr_names == ["ce/mean", "cp/mean", "cp/std"]

    h5_name = path.name
    items = root.findall(".//Attribute/DataItem")
    refs = sorted(it.text for it in items)
    assert refs == [
        f"{h5_name}:/ce/mean",
        f"{h5_name}:/cp/mean",
        f"{h5_name}:/cp/std",
    ]


def test_write_stats_xdmf_skips_meta_groups(tmp_path, triangles, vertices):
    path = tmp_path / "results.h5"
    write_stats_field(
        path,
        group="cp",
        stat_name="mean",
        values=np.array([0.1, 0.2]),
        triangles=triangles,
        vertices=vertices,
    )
    write_timeseries_meta(
        path, time_steps=np.array([0.0]), time_normalized=np.array([0.0])
    )

    xdmf_path = tmp_path / "results.xdmf"
    write_stats_xdmf(path, xdmf_path)

    tree = ET.parse(xdmf_path)
    attr_names = [a.attrib["Name"] for a in tree.getroot().findall(".//Attribute")]
    assert "meta" not in " ".join(attr_names)
    assert attr_names == ["cp/mean"]


def test_xdmf_files_have_doctype_header(tmp_path, timeseries_h5):
    xdmf_path = tmp_path / "ts.xdmf"
    write_temporal_xdmf(timeseries_h5, xdmf_path, group="pressure")
    text = pathlib.Path(xdmf_path).read_text()
    assert text.startswith('<?xml version="1.0" ?>\n<!DOCTYPE Xdmf SYSTEM "Xdmf.dtd">')
