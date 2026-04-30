"""Shared fixtures and helpers for ``tests/pressure``.

Centralises the small bits that were duplicated across multiple test
modules:

- Pressure-fixture file paths (building / galpao / repo-root big case).
- ``zoning_full`` and the ``make_*_cfg`` config builders so each test
  doesn't have to re-author 12 lines of ``CpCaseConfig(...)`` boilerplate.
- ``iter_stats_leaves`` to walk a stats H5 and pull the leaf groups +
  per-group stat names (used by both correctness and perf tests).
- Session-scoped ``building_cp_h5`` / ``galpao_cp_h5`` fixtures so
  multi-body / multi-case tests don't re-run Cp on every invocation.

Anything project-wide (paths, marks) lives here so swapping a fixture
takes one edit instead of ``grep BODY_H5`` across the suite.
"""

from __future__ import annotations

import pathlib
from collections.abc import Iterable

import h5py
import pytest

from cfdmod.io.geometry.transformation_config import TransformationConfig
from cfdmod.pressure import (
    BasicStatisticModel,
    BodyConfig,
    BodyDefinition,
    CeCaseConfig,
    CeConfig,
    CfCaseConfig,
    CfConfig,
    CmCaseConfig,
    CmConfig,
    CpCaseConfig,
    CpConfig,
    MomentBodyConfig,
    ParameterizedStatisticModel,
    ZoningModel,
    run_cp,
)
from cfdmod.pressure.parameters import (
    ExtremeAbsoluteParamsModel,
    ExtremeGumbelParamsModel,
    ExtremeMovingAverageParamsModel,
    ExtremePeakParamsModel,
    MeanEquivalentParamsModel,
    ZoningConfig,
)

# ---------------------------------------------------------------------------
# Fixture file paths
# ---------------------------------------------------------------------------

REPO_ROOT = pathlib.Path(__file__).resolve().parents[2]
FIXTURES_PRESSURE = REPO_ROOT / "fixtures" / "tests" / "pressure"

# Small XDMF fixtures (no LNAS surfaces -- single "all" surface from the H5).
BUILDING_BODY_H5 = FIXTURES_PRESSURE / "xdmf" / "bodies.building.h5"
BUILDING_PROBE_H5 = FIXTURES_PRESSURE / "xdmf" / "points.point0.h5"

# Galpao fixtures with authored LNAS surfaces (for multi-body tests).
GALPAO_BODY_H5 = FIXTURES_PRESSURE / "data" / "bodies.galpao.h5"
GALPAO_PROBE_H5 = FIXTURES_PRESSURE / "data" / "points.static_pressure.h5"
GALPAO_MESH = FIXTURES_PRESSURE / "galpao" / "galpao.normalized.lnas"
GALPAO_CP_TIMESERIES = FIXTURES_PRESSURE / "data" / "cp_t.normalized.h5"

# Real-world big-case fixtures at the repo root (perf tests only). May be
# absent in CI; perf tests skip when missing.
BIG_BODY_H5 = REPO_ROOT / "bodies.body_cp body.h5"
BIG_PROBE_H5 = REPO_ROOT / "points.point_cp ref.h5"


def big_case_available() -> bool:
    """True iff the repo-root big-case files are present."""
    return BIG_BODY_H5.exists() and BIG_PROBE_H5.exists()


# ---------------------------------------------------------------------------
# Builders
# ---------------------------------------------------------------------------


def zoning_full() -> ZoningModel:
    """A ZoningModel that puts every triangle in a single (-inf, +inf) region."""
    return ZoningModel(
        x_intervals=[float("-inf"), float("inf")],
        y_intervals=[float("-inf"), float("inf")],
        z_intervals=[float("-inf"), float("inf")],
    )


def basic_stats(*names: str) -> list[BasicStatisticModel]:
    """Shorthand for a list of BasicStatisticModel by name."""
    return [BasicStatisticModel(stats=n) for n in names]


def make_cp_cfg(
    *,
    label: str = "default",
    statistics: Iterable | None = None,
    timestep_range: tuple[float, float] = (0.0, 1e9),
    macroscopic_type: str = "pressure",
    reference_pressure: str = "probe",
    simul_U_H: float = 1.0,
    simul_characteristic_length: float = 1.0,
    fluid_density: float = 1.0,
) -> CpCaseConfig:
    """Build a single-label CpCaseConfig with sensible test defaults."""
    if statistics is None:
        statistics = basic_stats("mean")
    return CpCaseConfig(
        pressure_coefficient={
            label: CpConfig(
                statistics=list(statistics),
                timestep_range=timestep_range,
                macroscopic_type=macroscopic_type,
                reference_pressure=reference_pressure,
                simul_U_H=simul_U_H,
                simul_characteristic_length=simul_characteristic_length,
                fluid_density=fluid_density,
            )
        }
    )


def make_cf_cfg(
    *,
    label: str = "scan",
    bodies: dict[str, BodyDefinition] | None = None,
    body_configs: list[BodyConfig] | None = None,
    statistics: Iterable | None = None,
    directions: list[str] | None = None,
    nominal_area: float = 1.0,
) -> CfCaseConfig:
    if bodies is None:
        bodies = {"all": BodyDefinition(surfaces=[])}
    if body_configs is None:
        body_configs = [BodyConfig(name="all", sub_bodies=zoning_full())]
    if statistics is None:
        statistics = basic_stats("mean")
    if directions is None:
        directions = ["x"]
    return CfCaseConfig(
        bodies=bodies,
        force_coefficient={
            label: CfConfig(
                statistics=list(statistics),
                bodies=body_configs,
                directions=directions,
                nominal_area=nominal_area,
                transformation=TransformationConfig(),
            )
        },
    )


def make_cm_cfg(
    *,
    label: str = "scan",
    bodies: dict[str, BodyDefinition] | None = None,
    body_configs: list[MomentBodyConfig] | None = None,
    statistics: Iterable | None = None,
    directions: list[str] | None = None,
    nominal_volume: float = 1.0,
) -> CmCaseConfig:
    if bodies is None:
        bodies = {"all": BodyDefinition(surfaces=[])}
    if body_configs is None:
        body_configs = [
            MomentBodyConfig(name="all", sub_bodies=zoning_full())
        ]
    if statistics is None:
        statistics = basic_stats("mean")
    if directions is None:
        directions = ["x"]
    return CmCaseConfig(
        bodies=bodies,
        moment_coefficient={
            label: CmConfig(
                statistics=list(statistics),
                bodies=body_configs,
                directions=directions,
                nominal_volume=nominal_volume,
                transformation=TransformationConfig(),
            )
        },
    )


# ---------------------------------------------------------------------------
# Stats-h5 walker
# ---------------------------------------------------------------------------


def iter_stats_leaves(h5_path: pathlib.Path) -> list[tuple[str, int, list[str]]]:
    """Walk a stats.h5 and return (group_path, n_tri, stat_names) per leaf
    group with embedded mesh. Used by correctness and perf tests."""
    rows: list[tuple[str, int, list[str]]] = []
    with h5py.File(h5_path, "r") as f:

        def visit(name: str, obj) -> None:
            if not isinstance(obj, h5py.Group):
                return
            if "Triangles" not in obj or "Geometry" not in obj:
                return
            n_tri = obj["Triangles"].shape[0]
            stats = sorted(
                k
                for k in obj.keys()
                if k not in ("Triangles", "Geometry", "processing_metadata")
                and isinstance(obj[k], h5py.Dataset)
            )
            rows.append((name, n_tri, stats))

        f.visititems(visit)
    return rows


# ---------------------------------------------------------------------------
# Session-scoped run_cp fixtures (avoid re-running Cp for every test)
# ---------------------------------------------------------------------------


@pytest.fixture(scope="session")
def building_cp_h5(tmp_path_factory) -> pathlib.Path:
    """Run Cp once on the building fixture; reused across the session."""
    out = tmp_path_factory.mktemp("building_cp")
    run_cp(
        body_h5=BUILDING_BODY_H5,
        probe_h5=BUILDING_PROBE_H5,
        cfg_path=make_cp_cfg(),
        output=out,
    )
    return out / "cp.default.time_series.h5"


@pytest.fixture(scope="session")
def galpao_cp_h5(tmp_path_factory) -> pathlib.Path:
    """Run Cp once on the galpao fixture (with authored LNAS); reused."""
    out = tmp_path_factory.mktemp("galpao_cp")
    run_cp(
        body_h5=GALPAO_BODY_H5,
        probe_h5=GALPAO_PROBE_H5,
        cfg_path=make_cp_cfg(),
        output=out,
        mesh_path=GALPAO_MESH,
    )
    return out / "cp.default.time_series.h5"


# Make the helpers importable as ``from tests.pressure.conftest import …``
# without having to also import them into each test file individually.
__all__ = [
    "BIG_BODY_H5",
    "BIG_PROBE_H5",
    "BUILDING_BODY_H5",
    "BUILDING_PROBE_H5",
    "GALPAO_BODY_H5",
    "GALPAO_CP_TIMESERIES",
    "GALPAO_MESH",
    "GALPAO_PROBE_H5",
    "REPO_ROOT",
    "basic_stats",
    "big_case_available",
    "iter_stats_leaves",
    "make_cf_cfg",
    "make_cm_cfg",
    "make_cp_cfg",
    "zoning_full",
]


# Re-export for convenience inside test files.
_ = (
    ExtremeAbsoluteParamsModel,
    ExtremeGumbelParamsModel,
    ExtremeMovingAverageParamsModel,
    ExtremePeakParamsModel,
    MeanEquivalentParamsModel,
    ParameterizedStatisticModel,
    CeCaseConfig,
    CeConfig,
    ZoningConfig,
)
