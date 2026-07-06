"""Shared pytest configuration.

Auto-marks any test that does not already carry a ``unit`` /
``integration`` / ``perf`` marker as ``unit``, so the documented
selector ``pytest -m "unit or integration"`` actually runs the whole
suite (previously only the handful of explicitly-marked files ran, and
everything else was silently deselected). ``perf`` stays opt-in.

Also registers the hypothesis example-budget profiles used by the
``property``-marked tests (issue #141). The property tests deliberately do
*not* pin ``max_examples`` themselves, so the active profile governs the count:

- ``default`` -- 50 examples, used by every normal PR / local run.
- ``dev`` -- 15 examples, for a quick inner loop (``--hypothesis-profile dev``).
- ``ci`` -- 1000 examples, for the nightly job (``--hypothesis-profile ci``).

``deadline=None`` on all profiles avoids spurious timing failures on shared CI
runners; correctness, not per-example latency, is what these tests assert.
"""

from __future__ import annotations

from hypothesis import settings

settings.register_profile("default", max_examples=50, deadline=None)
settings.register_profile("dev", max_examples=15, deadline=None)
settings.register_profile("ci", max_examples=1000, deadline=None)
settings.load_profile("default")

_SELECTION_MARKERS = {"unit", "integration", "perf"}


def pytest_collection_modifyitems(config, items):
    for item in items:
        if not _SELECTION_MARKERS.intersection(m.name for m in item.iter_markers()):
            item.add_marker("unit")
