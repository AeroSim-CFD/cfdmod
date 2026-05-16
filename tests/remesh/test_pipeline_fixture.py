"""Integration + perf tests for the container pipeline on the remesh fixture.

The fixture (``fixtures/tests/remesh/``) is a 6-container, 100-timestep
subset of the SN container-pack dataset:

- 2 containers from the first connectivity component (cc0_c0, cc0_c1)
- 2 from the second (cc1_c0, cc1_c1)
- 2 from the last (cc30_c34, cc30_c35)

Tests drive the same in-memory pipeline as
``notebooks/regroup_containers.ipynb`` via ``run_container_pipeline`` in
``conftest.py``.
"""

from __future__ import annotations

import h5py
import numpy as np
import pytest
import yaml

from tests.remesh.conftest import (
    FIXTURE_BODY,
    FIXTURE_MANIFEST,
    FIXTURE_PROBE,
    run_container_pipeline,
)


@pytest.mark.integration
def test_pipeline_runs_on_remesh_fixture(tmp_path):
    """Full pipeline on the 6-container fixture: slice + remesh + Cp stream."""
    res = run_container_pipeline(FIXTURE_BODY, FIXTURE_PROBE, tmp_path)

    # The fixture has 6 containers across 3 connectivity components; after the
    # subset extraction the body H5 keeps 121 parent triangles.
    assert res.n_parents == 121
    assert res.n_fragments >= res.n_parents  # slicing only adds triangles
    # 6 containers x up to 6 axis-aligned faces, but many faces don't exist
    # (container fans only have triangles on the visible/external faces).
    assert 6 <= res.n_per_face_buckets <= 36
    # Coplanar merge: each surface collapses to ~2 triangles. Cap upper bound
    # generously to allow non-coplanar boundary cases.
    assert res.n_coarse_triangles >= 2 * res.n_per_face_buckets / 2  # at least 1 tri / bucket
    assert res.n_coarse_triangles <= 6 * res.n_per_face_buckets
    assert res.n_timesteps == 100


@pytest.mark.integration
def test_pipeline_outputs_are_finite_and_match_manifest(tmp_path):
    """Cp output has no NaN, geometry is non-degenerate, surfaces map 1:1 to buckets."""
    res = run_container_pipeline(FIXTURE_BODY, FIXTURE_PROBE, tmp_path)

    manifest = yaml.safe_load(FIXTURE_MANIFEST.read_text())
    assert manifest["selection"]["n_containers"] == 6
    assert manifest["timesteps"]["n_kept"] == 100
    assert len(manifest["selection"]["container_names"]) == 6

    # Surface name set on the remeshed mesh equals the per-face bucket count
    # (each named surface is one (container, axis-aligned face) bucket).
    surfaces = list(res.remeshed_lnas.surfaces.keys())
    assert len(surfaces) == res.n_per_face_buckets
    # Each surface name matches the "<cell>_<dir>" convention.
    for name in surfaces:
        assert any(name.endswith(f"_{d}") for d in ("xp", "xn", "yp", "yn", "zp", "zn"))

    with h5py.File(res.remeshed_h5, "r") as f:
        tkeys = [k for k in f["cp"].keys() if k.startswith("t")]
        assert len(tkeys) == 100
        first = f["cp"][tkeys[0]][:]
        last = f["cp"][tkeys[-1]][:]
        assert first.shape[0] == res.n_coarse_triangles
        assert np.isnan(first).sum() == 0
        assert np.isnan(last).sum() == 0
        # Sanity range: Cp values shouldn't be wildly off; the fixture's
        # absolute Cp scale comes from the unnormalised pressure data, so we
        # only assert finiteness + non-trivial variation.
        assert float(first.max()) > float(first.min())


@pytest.mark.perf
def test_pipeline_perf_on_remesh_fixture(tmp_path):
    """Wall-time budget for the fixture pipeline.

    The fixture is tiny (121 parent tris, 100 timesteps) so the whole
    pipeline should complete in well under a second. A regression in the
    slicer or remesher will blow past this budget long before it shows up
    on production-size meshes.
    """
    res = run_container_pipeline(FIXTURE_BODY, FIXTURE_PROBE, tmp_path)
    print(
        f"\npipeline timings (s): "
        f"load={res.timings['load_mesh']:.3f} "
        f"slice={res.timings['slice']:.3f} "
        f"remesh={res.timings['remesh']:.3f} "
        f"stream_cp={res.timings['stream_cp']:.3f} "
        f"total={res.timings['total']:.3f}"
    )
    # Generous budget; we want regressions, not flakiness.
    assert res.timings["total"] < 5.0
    assert res.timings["slice"] < 3.0
    assert res.timings["remesh"] < 1.0
    assert res.timings["stream_cp"] < 1.0
