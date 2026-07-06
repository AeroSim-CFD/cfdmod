"""Shared fixture paths for ``tests/remesh``.

The fixture under ``fixtures/tests/remesh/`` is a 6-container, 100-timestep
subset of the SN container-pack dataset (see ``manifest.yaml`` and
``build_fixture.py`` there for the extraction recipe). Test files use the
paths exposed here to point at it.
"""

from __future__ import annotations

import pathlib

FIXTURE_DIR = pathlib.Path(__file__).resolve().parents[2] / "fixtures" / "tests" / "remesh"
FIXTURE_BODY = FIXTURE_DIR / "bodies.h5"
FIXTURE_PROBE = FIXTURE_DIR / "points.h5"
FIXTURE_MANIFEST = FIXTURE_DIR / "manifest.yaml"

__all__ = [
    "FIXTURE_BODY",
    "FIXTURE_PROBE",
    "FIXTURE_MANIFEST",
]
