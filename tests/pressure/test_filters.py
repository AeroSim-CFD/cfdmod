"""Back-compat shim test: importing from cfdmod.pressure.filters still
works (with a DeprecationWarning) after the move to cfdmod.filters.

The full filter test suite lives in tests/filters/.
"""

from __future__ import annotations

import warnings

import pytest

pytestmark = pytest.mark.unit


def test_pressure_filters_shim_emits_deprecation_warning():
    # Re-importing a module after warnings.warn is one-shot; force a
    # fresh re-import so the warning fires deterministically.
    import importlib
    import sys

    sys.modules.pop("cfdmod.pressure.filters", None)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        importlib.import_module("cfdmod.pressure.filters")
    assert any(
        issubclass(w.category, DeprecationWarning) and "cfdmod.filters" in str(w.message)
        for w in caught
    ), "Expected a DeprecationWarning naming cfdmod.filters"


def test_shim_apply_filters_is_h5_wrapper():
    """The back-compat ``apply_filters`` from the shim must be the H5 wrapper
    (preserves the previous behaviour of the file-in / file-out call)."""
    from cfdmod.filters import apply_filters_h5
    from cfdmod.pressure.filters import apply_filters as shim_apply_filters

    assert shim_apply_filters is apply_filters_h5
