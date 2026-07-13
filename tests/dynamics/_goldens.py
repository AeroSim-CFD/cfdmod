"""Loader for frozen legacy-parity golden values.

The archive was captured from ``cfdmod.hfpi`` during the one-shot migration
(the throwaway generator has since been removed). Tests load these frozen
values instead of importing the legacy module, so the characterization guard
survives the cutover.
"""

from __future__ import annotations

import pathlib

import numpy as np

_PATH = (
    pathlib.Path(__file__).parents[2] / "fixtures" / "tests" / "dynamics" / "legacy_goldens.npz"
)
_CACHE: dict[str, np.ndarray] | None = None


def golden(key: str) -> np.ndarray:
    """Return the frozen legacy output stored under ``key``."""
    global _CACHE
    if _CACHE is None:
        with np.load(_PATH) as data:
            _CACHE = {k: data[k] for k in data.files}
    return _CACHE[key]
