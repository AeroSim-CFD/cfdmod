"""Integration tests for moment-coefficient case expansion in run.py."""

from __future__ import annotations

import pathlib
import tempfile

import h5py
import numpy as np

from cfdmod.io.geometry.transformation_config import TransformationConfig
from cfdmod.io.mesh import mesh_from_h5
from cfdmod.pressure import (
    BodyConfig,
    BodyDefinition,
    CmCaseConfig,
    CmConfig,
    CpCaseConfig,
    CpConfig,
    MomentBodyConfig,
    ZoningModel,
    run_cm,
    run_cp,
)
from cfdmod.pressure.parameters import (
    BasicStatisticModel,
)
from cfdmod.pressure.run import _bbox_corners_xy_cases, _expand_moment_cases
from cfdmod.utils import save_yaml

BODY_H5 = pathlib.Path("fixtures/tests/pressure/xdmf/bodies.building.h5")
PROBE_H5 = pathlib.Path("fixtures/tests/pressure/xdmf/points.point0.h5")


def _zoning_full() -> ZoningModel:
    return ZoningModel(
        x_intervals=[float("-inf"), float("inf")],
        y_intervals=[float("-inf"), float("inf")],
        z_intervals=[float("-inf"), float("inf")],
    )


def test_bbox_corners_xy_cases_yields_four_keys():
    mesh = mesh_from_h5(BODY_H5)
    body = MomentBodyConfig(name="b", sub_bodies=_zoning_full())
    cases = _bbox_corners_xy_cases(body, mesh, sfc_list=["all"])

    assert set(cases) == {"xmin_ymin", "xmin_ymax", "xmax_ymin", "xmax_ymax"}
    # Each case should have at least one region with an origin tuple.
    for label, region_origins in cases.items():
        assert region_origins, f"case {label} is empty"
        any_pt = next(iter(region_origins.values()))
        assert len(any_pt) == 3 and all(isinstance(c, float) for c in any_pt)

    # The four corners of the same region must share their min-z and span x and y.
    region_int = next(iter(cases["xmin_ymin"]))
    p_xmin_ymin = cases["xmin_ymin"][region_int]
    p_xmax_ymin = cases["xmax_ymin"][region_int]
    p_xmin_ymax = cases["xmin_ymax"][region_int]
    p_xmax_ymax = cases["xmax_ymax"][region_int]
    assert p_xmin_ymin[2] == p_xmax_ymin[2] == p_xmin_ymax[2] == p_xmax_ymax[2]
    assert p_xmin_ymin[0] < p_xmax_ymin[0]
    assert p_xmin_ymin[1] < p_xmin_ymax[1]


def test_expand_moment_cases_explicit_dict():
    mesh = mesh_from_h5(BODY_H5)
    body = MomentBodyConfig(
        name="b",
        sub_bodies=_zoning_full(),
        lever_origin_cases={
            "a": {0: (0.0, 0.0, 0.0)},
            "b": {0: (1.0, 1.0, 1.0)},
        },
    )
    cfg = CmConfig(
        statistics=[BasicStatisticModel(stats="mean")],
        bodies=[body],
        directions=["x"],
        transformation=TransformationConfig(),
    )
    bodies_def = {"b": BodyDefinition(surfaces=["all"])}
    runs = _expand_moment_cases(cfg, bodies_def, mesh)

    # One run per case, each carrying a single derived body with the case's origins.
    names = [run_cfg.bodies[0].name for run_cfg, _ in runs]
    assert names == ["b.a", "b.b"]
    for run_cfg, run_def in runs:
        assert len(run_cfg.bodies) == 1
        derived = run_cfg.bodies[0]
        assert derived.lever_strategy == "fixed"
        assert derived.lever_origin_cases is None
        assert derived.region_lever_origins is not None
        assert run_def[derived.name].surfaces == ["all"]


def test_expand_moment_cases_no_cases_passes_body_through():
    mesh = mesh_from_h5(BODY_H5)
    body = MomentBodyConfig(name="b", sub_bodies=_zoning_full(), lever_origin=(1.0, 2.0, 3.0))
    cfg = CmConfig(
        statistics=[BasicStatisticModel(stats="mean")],
        bodies=[body],
        directions=["x"],
        transformation=TransformationConfig(),
    )
    bodies_def = {"b": BodyDefinition(surfaces=["all"])}
    runs = _expand_moment_cases(cfg, bodies_def, mesh)
    assert len(runs) == 1
    run_cfg, _ = runs[0]
    assert run_cfg.bodies[0] is body


def test_run_cm_with_bbox_corners_writes_one_run_per_case(tmp_path):
    """End-to-end: 4 bbox-corner cases produce 4 timeseries files + 4 stats subgroups."""
    out = tmp_path
    save_yaml(
        CpCaseConfig(
            pressure_coefficient={
                "default": CpConfig(
                    statistics=[BasicStatisticModel(stats="mean")],
                    timestep_range=(0.0, 1e9),
                    macroscopic_type="pressure",
                    reference_pressure="average",
                    simul_U_H=1.0,
                    simul_characteristic_length=1.0,
                    fluid_density=1.0,
                )
            }
        ).model_dump(),
        out / "cp.yaml",
    )
    run_cp(
        body_h5=BODY_H5,
        probe_h5=PROBE_H5,
        cfg_path=out / "cp.yaml",
        output=out,
    )
    cp_h5 = out / "cp.default.time_series.h5"
    assert cp_h5.exists()

    cm_cfg = CmCaseConfig(
        bodies={"b": BodyDefinition(surfaces=["all"])},
        moment_coefficient={
            "case_scan": CmConfig(
                statistics=[BasicStatisticModel(stats="mean")],
                bodies=[
                    MomentBodyConfig(
                        name="b",
                        sub_bodies=_zoning_full(),
                        lever_strategy="region_bbox_corners_xy",
                    )
                ],
                directions=["x", "y"],
                transformation=TransformationConfig(),
            )
        },
    )
    save_yaml(cm_cfg.model_dump(), out / "cm.yaml")
    run_cm(cp_h5=cp_h5, cfg_path=out / "cm.yaml", output=out)

    # 4 corner cases -> 4 timeseries files
    ts_files = sorted(out.glob("Cm.case_scan.b.*.time_series.h5"))
    case_names = [p.stem.split(".time_series")[0].rsplit(".", 1)[-1] for p in ts_files]
    assert sorted(case_names) == ["xmax_ymax", "xmax_ymin", "xmin_ymax", "xmin_ymin"]

    # stats.h5 should have one subgroup per (direction, case)
    with h5py.File(out / "stats.h5", "r") as f:
        leaves = []
        def visit(name, obj):
            if isinstance(obj, h5py.Group) and "Triangles" in obj and "Geometry" in obj:
                leaves.append(name)
        f.visititems(visit)
    cm_leaves = {n for n in leaves if n.startswith("cm_")}
    expected = {
        f"cm_{d}/case_scan/b.{c}"
        for d in ("x", "y")
        for c in ("xmin_ymin", "xmin_ymax", "xmax_ymin", "xmax_ymax")
    }
    assert cm_leaves == expected
