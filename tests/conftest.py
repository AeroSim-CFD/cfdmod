"""Shared pytest configuration.

Auto-marks any test that does not already carry a ``unit`` /
``integration`` / ``perf`` marker as ``unit``, so the documented
selector ``pytest -m "unit or integration"`` actually runs the whole
suite (previously only the handful of explicitly-marked files ran, and
everything else was silently deselected). ``perf`` stays opt-in.
"""

from __future__ import annotations

_SELECTION_MARKERS = {"unit", "integration", "perf"}


def pytest_collection_modifyitems(config, items):
    for item in items:
        if not _SELECTION_MARKERS.intersection(m.name for m in item.iter_markers()):
            item.add_marker("unit")
