"""Shared fixture paths + a thin wrapper around :func:`cfdmod.remesh.run_container_pipeline`.

The full pipeline implementation lives in ``cfdmod.remesh.pipeline`` so the
notebook (``notebooks/regroup_containers.ipynb``) and the test suite both
drive the same code path. This module only exposes:

- the fixture file paths (``FIXTURE_BODY`` / ``FIXTURE_PROBE`` /
  ``FIXTURE_MANIFEST``),
- a backward-compatible ``run_container_pipeline`` wrapper that accepts the
  test-friendly per-parameter signature and forwards to the public function.
"""

from __future__ import annotations

import pathlib

from cfdmod.pressure.parameters import CpConfig
from cfdmod.remesh import PipelineResult
from cfdmod.remesh import run_container_pipeline as _run_public

FIXTURE_DIR = pathlib.Path(__file__).resolve().parents[2] / "fixtures" / "tests" / "remesh"
FIXTURE_BODY = FIXTURE_DIR / "bodies.h5"
FIXTURE_PROBE = FIXTURE_DIR / "points.h5"
FIXTURE_MANIFEST = FIXTURE_DIR / "manifest.yaml"

__all__ = [
    "FIXTURE_BODY",
    "FIXTURE_PROBE",
    "FIXTURE_MANIFEST",
    "PipelineResult",
    "run_container_pipeline",
]


def run_container_pipeline(
    body_h5: pathlib.Path,
    probe_h5: pathlib.Path,
    output_dir: pathlib.Path,
    *,
    target_size_x: float = 6.34,
    target_size_y: float = 2.58,
    target_size_z: float = 2.6,
    min_triangles: int = 4,
    simul_U_H: float = 1.0,
    fluid_density: float = 1.0,
    macroscopic_type: str = "pressure",
    reference_pressure: str = "probe",
) -> PipelineResult:
    """Test-friendly wrapper that builds a default ``CpConfig`` on the caller's
    behalf and delegates to :func:`cfdmod.remesh.run_container_pipeline`.
    """
    cp_config = CpConfig(
        timestep_range=(-1e30, 1e30),  # accept all timesteps in the body H5
        simul_U_H=simul_U_H,
        simul_characteristic_length=1.0,
        fluid_density=fluid_density,
        macroscopic_type=macroscopic_type,
        reference_pressure=reference_pressure,
    )
    return _run_public(
        body_h5=body_h5,
        probe_h5=probe_h5,
        output_dir=output_dir,
        cp_config=cp_config,
        target_size_x=target_size_x,
        target_size_y=target_size_y,
        target_size_z=target_size_z,
        min_triangles=min_triangles,
    )
