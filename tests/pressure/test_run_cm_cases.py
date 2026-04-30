"""Integration tests for moment-coefficient case expansion in run.py."""

from __future__ import annotations

import h5py

from cfdmod.io.mesh import mesh_from_h5
from cfdmod.pressure import (
    BodyDefinition,
    CmCaseConfig,
    CmConfig,
    MomentBodyConfig,
    run_cm,
)
from cfdmod.pressure.parameters import BasicStatisticModel
from cfdmod.pressure.run import _bbox_corners_xy_cases, _expand_moment_cases
from cfdmod.io.geometry.transformation_config import TransformationConfig
from tests.pressure.conftest import (
    BUILDING_BODY_H5,
    iter_stats_leaves,
    zoning_full,
)
import pytest

pytestmark = pytest.mark.integration


# ---------------------------------------------------------------------------
# Unit tests: case expansion helpers
# ---------------------------------------------------------------------------


def test_bbox_corners_xy_cases_yields_four_keys():
    mesh = mesh_from_h5(BUILDING_BODY_H5)
    body = MomentBodyConfig(name="b", sub_bodies=zoning_full())
    cases = _bbox_corners_xy_cases(body, mesh, sfc_list=["all"])

    assert set(cases) == {"xmin_ymin", "xmin_ymax", "xmax_ymin", "xmax_ymax"}
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
    mesh = mesh_from_h5(BUILDING_BODY_H5)
    body = MomentBodyConfig(
        name="b",
        sub_bodies=zoning_full(),
        lever_origin_cases={
            "a": {0: (0.0, 0.0, 0.0)},
            "b": {0: (1.0, 1.0, 1.0)},
        },
    )
    cfg = CmConfig(
        statistics=[BasicStatisticModel(stats="mean")],
        bodies=[body],
        directions=["x"],
        nominal_volume=1.0,
        transformation=TransformationConfig(),
    )
    bodies_def = {"b": BodyDefinition(surfaces=["all"])}
    runs = _expand_moment_cases(cfg, bodies_def, mesh)

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
    mesh = mesh_from_h5(BUILDING_BODY_H5)
    body = MomentBodyConfig(
        name="b", sub_bodies=zoning_full(), lever_origin=(1.0, 2.0, 3.0)
    )
    cfg = CmConfig(
        statistics=[BasicStatisticModel(stats="mean")],
        bodies=[body],
        directions=["x"],
        nominal_volume=1.0,
        transformation=TransformationConfig(),
    )
    bodies_def = {"b": BodyDefinition(surfaces=["all"])}
    runs = _expand_moment_cases(cfg, bodies_def, mesh)
    assert len(runs) == 1
    run_cfg, _ = runs[0]
    assert run_cfg.bodies[0] is body


# ---------------------------------------------------------------------------
# Integration: full run_cp -> run_cm with bbox corners
# ---------------------------------------------------------------------------


def test_run_cm_with_bbox_corners_writes_one_run_per_case(building_cp_h5, tmp_path):
    """4 corner cases -> 4 timeseries files + 4*N(direction) stats subgroups."""
    out = tmp_path
    # Stash a copy of cp.h5 next to where Cm output will land so the test
    # uses the session-scoped Cp without polluting the session tmp dir.
    cm_cfg = CmCaseConfig(
        bodies={"b": BodyDefinition(surfaces=["all"])},
        moment_coefficient={
            "case_scan": CmConfig(
                statistics=[BasicStatisticModel(stats="mean")],
                bodies=[
                    MomentBodyConfig(
                        name="b",
                        sub_bodies=zoning_full(),
                        lever_strategy="region_bbox_corners_xy",
                    )
                ],
                directions=["x", "y"],
                nominal_volume=1.0,
                transformation=TransformationConfig(),
            )
        },
    )
    run_cm(cp_h5=building_cp_h5, cfg_path=cm_cfg, output=out)

    ts_files = sorted(out.glob("Cm.case_scan.b.*.time_series.h5"))
    case_names = [p.stem.split(".time_series")[0].rsplit(".", 1)[-1] for p in ts_files]
    assert sorted(case_names) == ["xmax_ymax", "xmax_ymin", "xmin_ymax", "xmin_ymin"]

    cm_leaves = {
        name for name, *_ in iter_stats_leaves(out / "stats.h5") if name.startswith("cm_")
    }
    expected = {
        f"cm_{d}/case_scan/b.{c}"
        for d in ("x", "y")
        for c in ("xmin_ymin", "xmin_ymax", "xmax_ymin", "xmax_ymax")
    }
    assert cm_leaves == expected
