"""Regression tests for Cf/Cm with more than one body in a single config."""

from __future__ import annotations

import pytest

from cfdmod.io.geometry.transformation_config import TransformationConfig
from cfdmod.pressure import (
    BodyConfig,
    BodyDefinition,
    CfCaseConfig,
    CfConfig,
    CmCaseConfig,
    CmConfig,
    MomentBodyConfig,
    run_cf,
    run_cm,
    run_cp,
)
from tests.pressure.conftest import (
    GALPAO_BODY_H5,
    GALPAO_MESH,
    GALPAO_PROBE_H5,
    basic_stats,
    iter_stats_leaves,
    make_cp_cfg,
    zoning_full,
)

pytestmark = pytest.mark.integration


def _galpao_cp(tmp_path):
    """Local Cp runner: galpao multi-body tests need an LNAS-backed cp.h5,
    distinct from the building/galpao session fixtures so each test gets a
    fresh output dir to write Cf/Cm into."""
    run_cp(
        body_h5=GALPAO_BODY_H5,
        probe_h5=GALPAO_PROBE_H5,
        cfg_path=make_cp_cfg(),
        output=tmp_path,
        mesh_path=GALPAO_MESH,
    )
    return tmp_path / "cp.default.time_series.h5"


def test_run_cf_with_two_bodies(tmp_path):
    """Two bodies on disjoint surface lists in one Cf config -- catches the
    pandas-merge column-suffix bug from earlier in the refactor."""
    cp_h5 = _galpao_cp(tmp_path)
    cf_cfg = CfCaseConfig(
        bodies={
            "body_a": BodyDefinition(surfaces=["m1_yp"]),
            "body_b": BodyDefinition(surfaces=["m2_zp"]),
        },
        force_coefficient={
            "scan": CfConfig(
                statistics=basic_stats("mean"),
                bodies=[
                    BodyConfig(name="body_a", sub_bodies=zoning_full()),
                    BodyConfig(name="body_b", sub_bodies=zoning_full()),
                ],
                directions=["x"],
                nominal_area=1.0,
                transformation=TransformationConfig(),
            )
        },
    )
    run_cf(cp_h5=cp_h5, cfg_path=cf_cfg, output=tmp_path, mesh_path=GALPAO_MESH)

    leaves = {name for name, *_ in iter_stats_leaves(tmp_path / "stats.h5")}
    assert {"cf_x/scan/body_a", "cf_x/scan/body_b"} <= leaves


def test_run_cm_with_two_bodies(tmp_path):
    """Same shape as test_run_cf_with_two_bodies but on the moment side
    (exercises the in-place rx/ry/rz update path)."""
    cp_h5 = _galpao_cp(tmp_path)
    cm_cfg = CmCaseConfig(
        bodies={
            "body_a": BodyDefinition(surfaces=["m1_yp"]),
            "body_b": BodyDefinition(surfaces=["m2_zp"]),
        },
        moment_coefficient={
            "scan": CmConfig(
                statistics=basic_stats("mean"),
                bodies=[
                    MomentBodyConfig(
                        name="body_a", sub_bodies=zoning_full(), lever_origin=(0.0, 0.0, 0.0)
                    ),
                    MomentBodyConfig(
                        name="body_b", sub_bodies=zoning_full(), lever_origin=(1.0, 1.0, 1.0)
                    ),
                ],
                directions=["x"],
                nominal_volume=1.0,
                transformation=TransformationConfig(),
            )
        },
    )
    run_cm(cp_h5=cp_h5, cfg_path=cm_cfg, output=tmp_path, mesh_path=GALPAO_MESH)

    leaves = {name for name, *_ in iter_stats_leaves(tmp_path / "stats.h5")}
    assert {"cm_x/scan/body_a", "cm_x/scan/body_b"} <= leaves
