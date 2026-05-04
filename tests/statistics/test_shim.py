"""Back-compat shim test: importing from cfdmod.pressure.statistics_runner
still works (with a DeprecationWarning) after the move to cfdmod.statistics.
"""

from __future__ import annotations

import warnings

import pytest

pytestmark = pytest.mark.unit


def test_pressure_statistics_runner_shim_emits_deprecation_warning():
    import importlib
    import sys

    sys.modules.pop("cfdmod.pressure.statistics_runner", None)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        importlib.import_module("cfdmod.pressure.statistics_runner")
    assert any(
        issubclass(w.category, DeprecationWarning)
        and "cfdmod.statistics" in str(w.message)
        for w in caught
    ), "Expected a DeprecationWarning naming cfdmod.statistics"


def test_shim_calculate_statistics_from_h5_is_h5_wrapper():
    """The back-compat ``calculate_statistics_from_h5`` from the shim must be
    the same callable as the new ``apply_statistics_h5``."""
    from cfdmod.pressure.statistics_runner import calculate_statistics_from_h5
    from cfdmod.statistics import apply_statistics_h5

    assert calculate_statistics_from_h5 is apply_statistics_h5
