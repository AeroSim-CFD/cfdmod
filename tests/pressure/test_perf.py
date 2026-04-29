"""Performance smoke tests for the v2 pressure pipeline on synthetic big data.

The full ``run_cp -> run_cf -> run_cm`` chain runs against a body / probe
pair generated in-fixture (no out-of-tree files), so the benchmark is
reproducible on any machine. Two reference scales:

================  =============  ==============  =================
``CFDMOD_PERF_SCALE``  triangles  timesteps      typical purpose
================  =============  ==============  =================
``tiny``                  5 000          200    quick smoke (~10 s)
``medium`` (default)     30 000        2 000    standard benchmark
``extreme``             150 000       10 000    1/5x scale of  user's worst
                                                case; minutes to run
================  =============  ==============  =================

Default ``pytest`` runs ignore these tests (the ``perf`` marker is
excluded via ``addopts -m 'not perf'``); opt in with::

    pytest -m perf                                    # medium
    CFDMOD_PERF_SCALE=tiny    pytest -m perf -s       # quick smoke
    CFDMOD_PERF_SCALE=extreme pytest -m perf -s       # the long one

Tunables (environment variables):

- ``CFDMOD_PERF_SCALE`` -- picks a row from the table above.
- ``CFDMOD_PERF_N_TRI`` / ``CFDMOD_PERF_N_STEPS`` -- override individual
  dimensions on top of the scale.
- ``CFDMOD_PERF_{CP,CF,CM}_BUDGET_S`` -- override individual wall-time
  budgets in seconds.
- ``CFDMOD_PERF_RSS_BUDGET_MB`` -- peak-RSS budget in MiB.

Budgets default to the per-scale entries below and are deliberately loose
(3-5x typical observed) so the suite isn't flaky on slow runners.
``_measure()`` prints a one-line "[perf] {label}: Xs peak_rss_delta=Y MiB"
report, so ``pytest -m perf -s`` doubles as a quick benchmark.
"""

from __future__ import annotations

import math
import os
import pathlib
import resource
import time
from contextlib import contextmanager

import h5py
import numpy as np
import pytest

from cfdmod.io.geometry.transformation_config import TransformationConfig
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
from tests.pressure.conftest import (
    basic_stats,
    iter_stats_leaves,
    make_cp_cfg,
    zoning_full,
)

pytestmark = [pytest.mark.perf]


# ---------------------------------------------------------------------------
# Tunables
# ---------------------------------------------------------------------------


_SCALES: dict[str, dict[str, int]] = {
    "tiny":    {"n_tri":   5_000, "n_steps":    200},
    "medium":  {"n_tri":  30_000, "n_steps":  2_000},
    "extreme": {"n_tri": 150_000, "n_steps": 10_000},
}

_DEFAULT_BUDGETS_S: dict[str, dict[str, float]] = {
    # cp / cf / cm wall-time budgets per scale, in seconds.
    "tiny":    {"cp":   30,  "cf":    60, "cm":   120},
    "medium":  {"cp":  120,  "cf":   600, "cm":  1800},
    "extreme": {"cp": 1800,  "cf":  7200, "cm": 14400},
}

_SCALE = os.environ.get("CFDMOD_PERF_SCALE", "medium").lower()
if _SCALE not in _SCALES:
    raise ValueError(
        f"CFDMOD_PERF_SCALE={_SCALE!r} not one of {sorted(_SCALES)}"
    )

_N_TRI = int(os.environ.get("CFDMOD_PERF_N_TRI", _SCALES[_SCALE]["n_tri"]))
_N_STEPS = int(os.environ.get("CFDMOD_PERF_N_STEPS", _SCALES[_SCALE]["n_steps"]))

_BUDGETS = _DEFAULT_BUDGETS_S[_SCALE]
_BUDGET_CP_S = float(os.environ.get("CFDMOD_PERF_CP_BUDGET_S", _BUDGETS["cp"]))
_BUDGET_CF_S = float(os.environ.get("CFDMOD_PERF_CF_BUDGET_S", _BUDGETS["cf"]))
_BUDGET_CM_S = float(os.environ.get("CFDMOD_PERF_CM_BUDGET_S", _BUDGETS["cm"]))
_BUDGET_PEAK_RSS_MB = float(os.environ.get("CFDMOD_PERF_RSS_BUDGET_MB", "32768"))


# ---------------------------------------------------------------------------
# Wall-time + peak-RSS measurement
# ---------------------------------------------------------------------------


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
            f"[perf:{_SCALE}] {label}: {elapsed:.2f}s  "
            f"peak_rss_delta={delta:.0f} MiB  rss_after={rss_after:.0f} MiB"
        )


# ---------------------------------------------------------------------------
# Synthetic body/probe generation
# ---------------------------------------------------------------------------


def _grid_triangles(n_tri: int) -> tuple[np.ndarray, np.ndarray]:
    """Return (triangles, vertices) for a planar quad-grid mesh covering
    [0, 200] x [0, 200] x [0, 0]; truncated to the first ``n_tri`` triangles.

    Each quad cell yields 2 triangles. The grid is sized so the total meets
    or exceeds ``n_tri``. Vectorised so it scales to 150k+ triangles in well
    under a second.
    """
    cells = (n_tri + 1) // 2
    cols = int(math.ceil(math.sqrt(cells)))
    rows = int(math.ceil(cells / cols))
    xs = np.linspace(0.0, 200.0, cols + 1)
    ys = np.linspace(0.0, 200.0, rows + 1)
    xv, yv = np.meshgrid(xs, ys)
    vertices = np.column_stack(
        [xv.ravel(), yv.ravel(), np.zeros(xv.size, dtype=np.float64)]
    ).astype(np.float64)

    nx = cols + 1
    cell_idx = np.arange(rows * cols)
    r = cell_idx // cols
    c = cell_idx % cols
    base = r * nx + c
    tris_a = np.column_stack([base, base + 1, base + nx])
    tris_b = np.column_stack([base + 1, base + nx + 1, base + nx])
    triangles = np.concatenate(
        [tris_a.reshape(-1, 3), tris_b.reshape(-1, 3)], axis=0
    )[:n_tri].astype(np.int32)
    return triangles, vertices


def _write_synthetic_body(
    path: pathlib.Path, *, n_tri: int, n_steps: int, seed: int = 0
) -> None:
    """Build a body XDMF+H5 in the v2 layout with synthetic random pressure.

    Streams timesteps one at a time into the open file so peak memory stays
    O(n_tri) regardless of n_steps; this matters for the extreme scale where
    ``n_tri * n_steps`` is on the order of 1.5e9 floats (12 GB on disk).
    """
    triangles, vertices = _grid_triangles(n_tri)
    rng = np.random.default_rng(seed)
    times = np.linspace(100.0, 800.0, n_steps)

    with h5py.File(path, "w") as f:
        f.create_dataset("Triangles", data=triangles)
        f.create_dataset("Geometry", data=vertices)
        pressure = f.create_group("pressure")
        for t in times:
            pressure.create_dataset(
                f"t{t}", data=rng.standard_normal(n_tri).astype(np.float64)
            )
        meta = f.create_group("meta")
        meta.create_dataset("time_steps", data=times.astype(np.float64))
        meta.create_dataset("time_normalized", data=times.astype(np.float64))


def _write_synthetic_probe(
    path: pathlib.Path, *, n_steps: int, seed: int = 1
) -> None:
    """Build a single-point probe XDMF+H5 (trivial mesh + random pressure)."""
    rng = np.random.default_rng(seed)
    times = np.linspace(100.0, 800.0, n_steps)

    with h5py.File(path, "w") as f:
        f.create_dataset("Triangles", data=np.array([[0, 0, 0]], dtype=np.int32))
        f.create_dataset(
            "Geometry", data=np.array([[0.0, 0.0, 0.0]], dtype=np.float64)
        )
        pressure = f.create_group("pressure")
        for t in times:
            pressure.create_dataset(
                f"t{t}", data=rng.standard_normal(1).astype(np.float64)
            )
        meta = f.create_group("meta")
        meta.create_dataset("time_steps", data=times.astype(np.float64))
        meta.create_dataset("time_normalized", data=times.astype(np.float64))


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def synthetic_inputs(tmp_path_factory) -> tuple[pathlib.Path, pathlib.Path]:
    """Generate a body + probe pair sized via env vars; reused module-wide."""
    out = tmp_path_factory.mktemp("perf_synth")
    body = out / "body.h5"
    probe = out / "probe.h5"
    with _measure(f"synthesise_inputs (n_tri={_N_TRI}, n_steps={_N_STEPS})"):
        _write_synthetic_body(body, n_tri=_N_TRI, n_steps=_N_STEPS)
        _write_synthetic_probe(probe, n_steps=_N_STEPS)
    return body, probe


@pytest.fixture(scope="module")
def synthetic_cp_h5(synthetic_inputs, tmp_path_factory) -> pathlib.Path:
    """Run Cp once on the synthetic body + probe, reused across this module."""
    body, probe = synthetic_inputs
    out = tmp_path_factory.mktemp("perf_cp")
    cfg = make_cp_cfg(
        timestep_range=(100.0, 800.0),
        macroscopic_type="rho",
        simul_U_H=1.0,
        simul_characteristic_length=10.0,
    )
    rss_before = _max_rss_mb()
    t0 = time.perf_counter()
    with _measure("run_cp"):
        run_cp(body_h5=body, probe_h5=probe, cfg_path=cfg, output=out)
    elapsed = time.perf_counter() - t0
    rss_delta = max(0.0, _max_rss_mb() - rss_before)
    cp_h5 = out / "cp.default.time_series.h5"
    assert cp_h5.exists()
    assert elapsed < _BUDGET_CP_S, f"run_cp took {elapsed:.1f}s > budget {_BUDGET_CP_S}s"
    assert rss_delta < _BUDGET_PEAK_RSS_MB
    return cp_h5


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


def test_perf_run_cp_shape_and_metadata(synthetic_cp_h5):
    """run_cp itself was timed by the fixture; here we assert the resulting
    file shape matches the synthesised mesh + the embedded metadata is set."""
    with h5py.File(synthetic_cp_h5, "r") as f:
        n_tri = f["Triangles"].shape[0]
        n_steps = len(f["cp"].keys())
    assert n_tri == _N_TRI
    assert n_steps == _N_STEPS

    md = read_processing_metadata(synthetic_cp_h5, "/")
    assert md["coefficient"] == "cp"


def test_perf_run_cf(synthetic_cp_h5, tmp_path):
    """Cf x/y/z on the synthetic body."""
    cf_cfg = CfCaseConfig(
        bodies={"pack": BodyDefinition(surfaces=[])},
        force_coefficient={
            "containers": CfConfig(
                statistics=basic_stats("mean", "rms"),
                bodies=[BodyConfig(name="pack", sub_bodies=zoning_full())],
                directions=["x", "y", "z"],
                transformation=TransformationConfig(),
            )
        },
    )

    t0 = time.perf_counter()
    rss_before = _max_rss_mb()
    with _measure("run_cf"):
        run_cf(cp_h5=synthetic_cp_h5, cfg_path=cf_cfg, output=tmp_path)
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


def test_perf_run_cm_with_corner_scan(synthetic_cp_h5, tmp_path):
    """Cm with the four-corner overturning-moment scan -- the heaviest of
    the three runs (4 independent runs per body, per direction)."""
    cm_cfg = CmCaseConfig(
        bodies={"pack": BodyDefinition(surfaces=[])},
        moment_coefficient={
            "containers": CmConfig(
                statistics=basic_stats("mean", "rms"),
                bodies=[
                    MomentBodyConfig(
                        name="pack",
                        sub_bodies=zoning_full(),
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
        run_cm(cp_h5=synthetic_cp_h5, cfg_path=cm_cfg, output=tmp_path)
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


def test_perf_inputs_unchanged(synthetic_inputs, synthetic_cp_h5):
    """The no-mutation guarantee on real-world H5 files -- run_cp must not
    add a /cp group (or anything else) into its body / probe inputs."""
    body, probe = synthetic_inputs
    with h5py.File(body, "r") as f:
        assert "cp" not in list(f.keys())
    with h5py.File(probe, "r") as f:
        assert "cp" not in list(f.keys())
