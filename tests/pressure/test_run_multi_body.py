"""Regression tests for Cf/Cm with more than one body in a single config."""

from __future__ import annotations

import pathlib

import h5py
import numpy as np
import pytest

from cfdmod.io.geometry.transformation_config import TransformationConfig
from cfdmod.pressure import (
    BasicStatisticModel,
    BodyConfig,
    BodyDefinition,
    CfCaseConfig,
    CfConfig,
    CmCaseConfig,
    CmConfig,
    CpCaseConfig,
    CpConfig,
    MomentBodyConfig,
    ZoningModel,
    run_cf,
    run_cm,
    run_cp,
)

BODY_H5 = pathlib.Path("fixtures/tests/pressure/data/bodies.galpao.h5")
PROBE_H5 = pathlib.Path("fixtures/tests/pressure/data/points.static_pressure.h5")
MESH_PATH = pathlib.Path("fixtures/tests/pressure/galpao/galpao.normalized.lnas")


def _zoning_full() -> ZoningModel:
    return ZoningModel(
        x_intervals=[float("-inf"), float("inf")],
        y_intervals=[float("-inf"), float("inf")],
        z_intervals=[float("-inf"), float("inf")],
    )


@pytest.fixture()
def cp_h5(tmp_path):
    """Run Cp once so the multi-body Cf/Cm tests share a single cp.h5."""
    cp_cfg = CpCaseConfig(
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
    )
    run_cp(
        body_h5=BODY_H5,
        probe_h5=PROBE_H5,
        cfg_path=cp_cfg,
        output=tmp_path,
        mesh_path=MESH_PATH,
    )
    yield tmp_path / "cp.default.time_series.h5"


def test_run_cf_with_two_bodies(cp_h5):
    """Two bodies on disjoint surface lists in one Cf config -- catches the
    pandas-merge column-suffix bug that fired once the second body was
    processed."""
    out = cp_h5.parent
    cf_cfg = CfCaseConfig(
        bodies={
            "body_a": BodyDefinition(surfaces=["m1_yp"]),
            "body_b": BodyDefinition(surfaces=["m2_zp"]),
        },
        force_coefficient={
            "scan": CfConfig(
                statistics=[BasicStatisticModel(stats="mean")],
                bodies=[
                    BodyConfig(name="body_a", sub_bodies=_zoning_full()),
                    BodyConfig(name="body_b", sub_bodies=_zoning_full()),
                ],
                directions=["x"],
                transformation=TransformationConfig(),
            )
        },
    )
    run_cf(cp_h5=cp_h5, cfg_path=cf_cfg, output=out, mesh_path=MESH_PATH)

    with h5py.File(out / "stats.h5", "r") as f:
        names = []
        def visit(name, obj):
            if isinstance(obj, h5py.Group) and "Triangles" in obj and "Geometry" in obj:
                names.append(name)
        f.visititems(visit)

    assert "cf_x/scan/body_a" in names
    assert "cf_x/scan/body_b" in names


def test_run_cm_with_two_bodies(cp_h5):
    """Same as test_run_cf_with_two_bodies but on the moment-coefficient side
    (which exercises the in-place rx/ry/rz update path)."""
    out = cp_h5.parent
    cm_cfg = CmCaseConfig(
        bodies={
            "body_a": BodyDefinition(surfaces=["m1_yp"]),
            "body_b": BodyDefinition(surfaces=["m2_zp"]),
        },
        moment_coefficient={
            "scan": CmConfig(
                statistics=[BasicStatisticModel(stats="mean")],
                bodies=[
                    MomentBodyConfig(
                        name="body_a", sub_bodies=_zoning_full(), lever_origin=(0.0, 0.0, 0.0)
                    ),
                    MomentBodyConfig(
                        name="body_b", sub_bodies=_zoning_full(), lever_origin=(1.0, 1.0, 1.0)
                    ),
                ],
                directions=["x"],
                transformation=TransformationConfig(),
            )
        },
    )
    run_cm(cp_h5=cp_h5, cfg_path=cm_cfg, output=out, mesh_path=MESH_PATH)

    with h5py.File(out / "stats.h5", "r") as f:
        names = []
        def visit(name, obj):
            if isinstance(obj, h5py.Group) and "Triangles" in obj and "Geometry" in obj:
                names.append(name)
        f.visititems(visit)

    assert "cm_x/scan/body_a" in names
    assert "cm_x/scan/body_b" in names
