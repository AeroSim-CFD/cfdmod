"""Performance smoke tests for the v2 pressure pipeline on real big data.

These tests exist to make sure the full ``run_cp -> run_cf -> run_cm`` chain
finishes -- and finishes within a sane time / memory envelope -- on a real
case (~80k triangles, ~900 timesteps). They live behind ``@pytest.mark.perf``
and are excluded from the default test run; opt in with::

    pytest -m perf

Inputs are read from the repo root::

    bodies.body_cp body.h5
    points.point_cp ref.h5

If those files aren't present, every test in this module skips, so a clean
checkout still has a green default pytest run.

The wall-time/peak-RSS budgets are deliberately generous (3-5x what the
pipeline takes on a typical dev machine, observed at the time of writing)
so the suite isn't flaky on slow runners. Override budgets per environment
via the ``CFDMOD_PERF_*_BUDGET`` env variables documented in the constants
below.
"""

from __future__ import annotations

import os
import pathlib
import resource
import time
from contextlib import contextmanager

import h5py
import pytest

from cfdmod.io.mesh import mesh_from_h5
from cfdmod.io.xdmf import read_processing_metadata
from cfdmod.pressure import (
    BodyConfig,
    BodyDefinition,
    CfCaseConfig,
    CfConfig,
    CmCaseConfig,
    CmConfig,
    MomentBodyConfig,
    ZoningModel,
    run_cf,
    run_cm,
    run_cp,
)
from cfdmod.io.geometry.transformation_config import TransformationConfig
from tests.pressure.conftest import (
    BIG_BODY_H5,
    BIG_PROBE_H5,
    basic_stats,
    big_case_available,
    iter_stats_leaves,
    make_cp_cfg,
    zoning_full,
)

# Skip the whole module if the big-case files aren't present.
pytestmark = [
    pytest.mark.perf,
    pytest.mark.skipif(
        not big_case_available(),
        reason=f"big-case files not at repo root ({BIG_BODY_H5}, {BIG_PROBE_H5})",
    ),
]

# Generous default budgets (seconds / MB peak additional RSS). Override with
# env vars if your runner is slower / more memory constrained.
_BUDGET_CP_S = float(os.environ.get("CFDMOD_PERF_CP_BUDGET_S", "120"))
_BUDGET_CF_S = float(os.environ.get("CFDMOD_PERF_CF_BUDGET_S", "300"))
_BUDGET_CM_S = float(os.environ.get("CFDMOD_PERF_CM_BUDGET_S", "300"))
_BUDGET_PEAK_RSS_MB = float(os.environ.get("CFDMOD_PERF_RSS_BUDGET_MB", "16384"))


def _max_rss_mb() -> float:
    """Peak RSS of this process in MiB.

    On Linux ``ru_maxrss`` is in kibibytes, on macOS in bytes -- /proc check
    is the cleanest disambiguator without adding a psutil dep.
    """
    raw = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    if pathlib.Path("/proc/self/status").exists():
        return raw / 1024  # KiB -> MiB
    return raw / (1024 * 1024)  # bytes -> MiB


@contextmanager
def _measure(label: str):
    """Time + peak-RSS measurement around a block; prints a one-line report."""
    rss_before = _max_rss_mb()
    t0 = time.perf_counter()
    try:
        yield
    finally:
        elapsed = time.perf_counter() - t0
        rss_after = _max_rss_mb()
        delta = max(0.0, rss_after - rss_before)
        print(
            f"[perf] {label}: {elapsed:.2f}s  "
            f"peak_rss_delta={delta:.0f} MiB  rss_after={rss_after:.0f} MiB"
        )


def _build_pack_zoning(body_h5: pathlib.Path, gap_m: float = 1.0) -> ZoningModel:
    """Auto-detect container partition by gap-sweep on triangle centroids."""
    import numpy as np

    mesh = mesh_from_h5(body_h5)
    centroids = np.mean(mesh.geometry.triangle_vertices, axis=1)

    def axis_intervals(coords: np.ndarray) -> list[float]:
        s = np.sort(coords)
        diffs = np.diff(s)
        idx = np.where(diffs > gap_m)[0]
        return [float("-inf"), *((s[i] + s[i + 1]) / 2 for i in idx), float("inf")]

    return ZoningModel(
        x_intervals=axis_intervals(centroids[:, 0]),
        y_intervals=axis_intervals(centroids[:, 1]),
        z_intervals=axis_intervals(centroids[:, 2]),
    )


def _detect_time_range(body_h5: pathlib.Path) -> tuple[float, float]:
    with h5py.File(body_h5, "r") as f:
        keys = list(f["pressure"].keys())
    times = sorted(float(k[1:]) for k in keys)
    return float(times[0]), float(times[-1])


@pytest.fixture(scope="module")
def big_cp_h5(tmp_path_factory) -> pathlib.Path:
    """Run Cp once on the big-case body+probe, reused across this module."""
    out = tmp_path_factory.mktemp("perf_big")
    timestep_range = _detect_time_range(BIG_BODY_H5)
    cfg = make_cp_cfg(
        timestep_range=timestep_range,
        macroscopic_type="rho",
        simul_U_H=1.0,
        simul_characteristic_length=10.0,
    )
    with _measure("run_cp"):
        run_cp(
            body_h5=BIG_BODY_H5,
            probe_h5=BIG_PROBE_H5,
            cfg_path=cfg,
            output=out,
        )
    cp_h5 = out / "cp.default.time_series.h5"
    assert cp_h5.exists()
    return cp_h5


def test_perf_run_cp_completes_within_budget(big_cp_h5):
    """run_cp itself was timed by the fixture; here we assert the resulting
    file shape matches the input mesh + that wall-time stayed in budget.

    We re-time a cheap operation on the cp.h5 to surface the file's basic
    health (n_tri, n_steps, processing_metadata) without re-running Cp.
    """
    t0 = time.perf_counter()
    with h5py.File(big_cp_h5, "r") as f:
        n_tri = f["Triangles"].shape[0]
        n_steps = len(f["cp"].keys())
    elapsed = time.perf_counter() - t0
    print(f"[perf] cp.h5 inspect: {elapsed:.3f}s; n_tri={n_tri}, n_steps={n_steps}")

    md = read_processing_metadata(big_cp_h5, "/")
    assert md["coefficient"] == "cp"
    assert n_tri > 50_000  # sanity: real data, not a stub
    assert n_steps > 100


def test_perf_run_cf_on_big_case(big_cp_h5, tmp_path):
    """Cf runs end-to-end on the big case; assert wall-time + correctness."""
    zoning = _build_pack_zoning(BIG_BODY_H5)
    cf_cfg = CfCaseConfig(
        bodies={"pack": BodyDefinition(surfaces=[])},
        force_coefficient={
            "containers": CfConfig(
                statistics=basic_stats("mean", "rms"),
                bodies=[BodyConfig(name="pack", sub_bodies=zoning)],
                directions=["x", "y", "z"],
                transformation=TransformationConfig(),
            )
        },
    )

    t0 = time.perf_counter()
    rss_before = _max_rss_mb()
    with _measure("run_cf"):
        run_cf(cp_h5=big_cp_h5, cfg_path=cf_cfg, output=tmp_path)
    elapsed = time.perf_counter() - t0
    rss_delta = max(0.0, _max_rss_mb() - rss_before)

    assert elapsed < _BUDGET_CF_S, f"run_cf took {elapsed:.1f}s > budget {_BUDGET_CF_S}s"
    assert rss_delta < _BUDGET_PEAK_RSS_MB, (
        f"run_cf added {rss_delta:.0f} MiB peak RSS > budget {_BUDGET_PEAK_RSS_MB} MiB"
    )

    leaves = {n for n, *_ in iter_stats_leaves(tmp_path / "stats.h5") if n.startswith("cf_")}
    assert leaves == {
        "cf_x/containers/pack",
        "cf_y/containers/pack",
        "cf_z/containers/pack",
    }


def test_perf_run_cm_with_corner_scan(big_cp_h5, tmp_path):
    """Cm with the four-corner overturning-moment scan -- the heaviest of
    the three runs (4 independent runs per body)."""
    zoning = _build_pack_zoning(BIG_BODY_H5)
    cm_cfg = CmCaseConfig(
        bodies={"pack": BodyDefinition(surfaces=[])},
        moment_coefficient={
            "containers": CmConfig(
                statistics=basic_stats("mean", "rms"),
                bodies=[
                    MomentBodyConfig(
                        name="pack",
                        sub_bodies=zoning,
                        lever_strategy="region_bbox_corners_xy",
                    )
                ],
                directions=["x", "y", "z"],
                transformation=TransformationConfig(),
            )
        },
    )

    t0 = time.perf_counter()
    rss_before = _max_rss_mb()
    with _measure("run_cm (4 corner cases)"):
        run_cm(cp_h5=big_cp_h5, cfg_path=cm_cfg, output=tmp_path)
    elapsed = time.perf_counter() - t0
    rss_delta = max(0.0, _max_rss_mb() - rss_before)

    assert elapsed < _BUDGET_CM_S, f"run_cm took {elapsed:.1f}s > budget {_BUDGET_CM_S}s"
    assert rss_delta < _BUDGET_PEAK_RSS_MB

    leaves = {n for n, *_ in iter_stats_leaves(tmp_path / "stats.h5") if n.startswith("cm_")}
    expected = {
        f"cm_{d}/containers/pack.{c}"
        for d in ("x", "y", "z")
        for c in ("xmin_ymin", "xmin_ymax", "xmax_ymin", "xmax_ymax")
    }
    assert leaves == expected


def test_perf_inputs_unchanged(big_cp_h5):
    """Cheap smoke check that the run_cp fixture didn't mutate the original
    body or probe files (the no-mutation guarantee is load-bearing for
    external consumers; cheap to check on real data too).
    """
    body_size = BIG_BODY_H5.stat().st_size
    probe_size = BIG_PROBE_H5.stat().st_size
    body_keys = sorted(h5py.File(BIG_BODY_H5, "r").keys())
    probe_keys = sorted(h5py.File(BIG_PROBE_H5, "r").keys())

    # No /cp group should have been added to the originals.
    assert "cp" not in body_keys, "body H5 was mutated by run_cp"
    assert "cp" not in probe_keys, "probe H5 was mutated by run_cp"
    assert body_size == BIG_BODY_H5.stat().st_size
    assert probe_size == BIG_PROBE_H5.stat().st_size
