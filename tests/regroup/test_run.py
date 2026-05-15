"""End-to-end tests for ``cfdmod.regroup.run``."""

from __future__ import annotations

import pathlib

import h5py
import numpy as np
import pytest
from lnas import LnasFormat

from cfdmod.geometry.grouping import (
    ByConnectivityGrouping,
    ByDivisionsGrouping,
    ByZoningGrouping,
)
from cfdmod.io.xdmf import read_processing_metadata, read_timeseries_meta
from cfdmod.regroup.parameters import (
    BySizeRoundedPerComponent,
    RegroupConfig,
)
from cfdmod.regroup.run import expand_regroup_chain, run_regroup

from tests.regroup.conftest import GALPAO_CP_H5, make_synthetic_cp_h5


def test_expand_per_component_rounded_division_count(two_container_mesh):
    chain = [
        ByConnectivityGrouping(name_template="cc{idx}", min_triangles=4),
        BySizeRoundedPerComponent(
            target_size_x=2.0,
            target_size_y=3.0,
            name_template="{parent}_r{idx}",
        ),
    ]
    expanded, consumed, parent_intervals, parent_triangles = expand_regroup_chain(
        chain, two_container_mesh, transformation=None
    )
    assert set(parent_intervals.keys()) == {"cc0", "cc1"}
    assert set(parent_triangles.keys()) == {"cc0", "cc1"}
    # First spec preserved; one ByDivisionsGrouping per component appended;
    # both connectivity components consumed (they're scaffolding).
    assert isinstance(expanded[0], ByConnectivityGrouping)
    assert consumed == {"cc0", "cc1"}
    div_specs = [s for s in expanded if isinstance(s, ByDivisionsGrouping)]
    assert len(div_specs) == 2
    # The larger 4x6 container's bbox of triangle centroids spans roughly
    # ~3.5 x ~5.5 -> n=2 along x, n=2 along y at the chosen targets.
    # The smaller 2x3 container spans roughly ~1.5 x ~2.5 -> n=1 along both.
    n_x_counts = sorted(s.n_div_x for s in div_specs)
    n_y_counts = sorted(s.n_div_y for s in div_specs)
    assert n_x_counts == [1, 2]
    assert n_y_counts == [1, 2]


def test_run_regroup_end_to_end_synthetic(two_container_mesh, tmp_path):
    in_h5 = tmp_path / "in.h5"
    n_tri = two_container_mesh.geometry.triangles.shape[0]
    make_synthetic_cp_h5(in_h5, n_triangles=n_tri, n_steps=2, seed=7)

    cfg = RegroupConfig(
        groupings=[
            ByConnectivityGrouping(name_template="cc{idx}", min_triangles=4),
            BySizeRoundedPerComponent(
                target_size_x=2.0,
                target_size_y=3.0,
                name_template="{parent}_r{idx}",
            ),
        ],
        aggregation="area_weighted_mean",
        timeseries_group="cp",
    )
    out_dir = tmp_path / "out"
    run_regroup(cfg, two_container_mesh, in_h5, out_dir)

    out_lnas = out_dir / "geometry.lnas"
    out_h5 = out_dir / "cp.regrouped.h5"
    out_xdmf = out_dir / "cp.regrouped.xdmf"
    assert out_lnas.exists()
    assert out_h5.exists()
    assert out_xdmf.exists()

    new = LnasFormat.from_file(out_lnas)
    # Five leaf regions: cc0 -> 2x2=4 cells; cc1 -> 1x1=1 cell.
    # The cc0/cc1 parent components are dropped as consumed scaffolding.
    assert len(new.surfaces) == 5
    assert "cc0" not in new.surfaces and "cc1" not in new.surfaces
    # All triangles are accounted for in exactly one surface (sum == n_tri).
    total = sum(int(arr.size) for arr in new.surfaces.values())
    assert total == new.geometry.triangles.shape[0]

    with h5py.File(out_h5, "r") as f:
        keys = sorted(k for k in f["cp"].keys() if k.startswith("t"))
    assert len(keys) == 2

    meta = read_timeseries_meta(out_h5)
    assert "region_labels" in meta
    assert len(meta["region_labels"]) == new.geometry.triangles.shape[0]

    md = read_processing_metadata(out_h5, "cp")
    assert md["config"]["regroup"]["aggregation"] == "area_weighted_mean"


def test_run_regroup_per_triangle_preserves_total_columns(small_mesh, tmp_path):
    in_h5 = tmp_path / "in.h5"
    make_synthetic_cp_h5(in_h5, n_triangles=8, n_steps=2)

    cfg = RegroupConfig(
        groupings=[
            ByZoningGrouping(
                x_intervals=[0.0, 1.0, 2.001], name_template="r{idx}"
            )
        ],
        aggregation="per_triangle",
        timeseries_group="cp",
    )
    out_dir = tmp_path / "out"
    run_regroup(cfg, small_mesh, in_h5, out_dir)

    with h5py.File(out_dir / "cp.regrouped.h5", "r") as f:
        first_key = sorted(k for k in f["cp"].keys() if k.startswith("t"))[0]
        out_col = f["cp"][first_key][:]
    # 2x2 grid = 8 triangles, all inside the zoning.
    assert out_col.shape == (8,)


def test_run_regroup_sliced_two_container(two_container_mesh, tmp_path):
    """Sliced mode on the two-container fixture: fragments inherit parent ids,
    output surface count matches the leaf cells, and HDF5 columns line up."""
    in_h5 = tmp_path / "in.h5"
    n_tri = two_container_mesh.geometry.triangles.shape[0]
    make_synthetic_cp_h5(in_h5, n_triangles=n_tri, n_steps=2, seed=11)

    cfg = RegroupConfig(
        groupings=[
            ByConnectivityGrouping(name_template="cc{idx}", min_triangles=4),
            BySizeRoundedPerComponent(
                target_size_x=2.0,
                target_size_y=3.0,
                name_template="{parent}_r{idx}",
            ),
        ],
        aggregation="sliced",
        timeseries_group="cp",
    )
    out_dir = tmp_path / "out"
    run_regroup(cfg, two_container_mesh, in_h5, out_dir)

    new = LnasFormat.from_file(out_dir / "geometry.lnas")
    # Same 5 leaf surfaces as the un-sliced run, but more triangles (fragments).
    assert len(new.surfaces) == 5
    assert new.geometry.triangles.shape[0] >= n_tri  # fragments only multiply

    with h5py.File(out_dir / "cp.regrouped.h5", "r") as f:
        keys = sorted(k for k in f["cp"].keys() if k.startswith("t"))
        assert len(keys) == 2
        n_cols = f["cp"][keys[0]].shape[0]
    # Cardinality match: one HDF5 column per output triangle.
    assert n_cols == new.geometry.triangles.shape[0]


@pytest.mark.skipif(
    not GALPAO_CP_H5.exists(), reason="galpao Cp fixture not available"
)
def test_run_regroup_on_galpao_fixture(tmp_path):
    """Sanity-check on the real Cp fixture: zoning + per_triangle round-trip."""
    from cfdmod.io.mesh import mesh_from_h5

    mesh = mesh_from_h5(GALPAO_CP_H5)
    n_tri = mesh.geometry.triangles.shape[0]
    assert n_tri > 0

    # Bbox-spanning zoning that splits the mesh into 4 cells.
    bbox_lo = mesh.geometry.vertices.min(axis=0)
    bbox_hi = mesh.geometry.vertices.max(axis=0)
    cfg = RegroupConfig(
        groupings=[
            ByZoningGrouping(
                x_intervals=[
                    float(bbox_lo[0]) - 1.0,
                    float((bbox_lo[0] + bbox_hi[0]) / 2),
                    float(bbox_hi[0]) + 1.0,
                ],
                y_intervals=[
                    float(bbox_lo[1]) - 1.0,
                    float((bbox_lo[1] + bbox_hi[1]) / 2),
                    float(bbox_hi[1]) + 1.0,
                ],
                name_template="r{idx}",
            )
        ],
        aggregation="area_weighted_mean",
        timeseries_group="cp",
    )

    out_dir = tmp_path / "galpao_out"
    run_regroup(cfg, GALPAO_CP_H5, GALPAO_CP_H5, out_dir)

    assert (out_dir / "geometry.lnas").exists()
    assert (out_dir / "cp.regrouped.h5").exists()
    assert (out_dir / "cp.regrouped.xdmf").exists()
