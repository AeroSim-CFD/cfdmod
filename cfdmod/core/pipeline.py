"""Pipeline = function composition over :class:`DataSource`.

A pipeline is just a callable. There is no class hierarchy, no
``apply`` method. Recipes build a pipeline by composing op functions
with their parameters bound via :func:`functools.partial`; the
resulting callable takes a :class:`DataSource` and returns the
transformed :class:`DataSource`.

This keeps the design honest: every step in a pipeline is a pure
function. There is no hidden state, no setup / teardown, no surprises
across runs.
"""

from __future__ import annotations

__all__ = [
    "Pipeline",
    "compose",
    "identity",
]

from functools import reduce
from typing import Callable

from cfdmod.core.data_source import DataSource

Pipeline = Callable[[DataSource], DataSource]
"""A pipeline is just a single-arg callable on :class:`DataSource`."""


def identity(ds: DataSource) -> DataSource:
    """Identity pipeline. Returns its input unchanged."""
    return ds


def compose(*ops: Callable[[DataSource], DataSource]) -> Pipeline:
    """Compose a sequence of single-arg callables left-to-right.

    The first op runs first; the last op's output is the pipeline's
    output. With no ops, returns :func:`identity`.
    """
    if not ops:
        return identity

    def run(ds: DataSource) -> DataSource:
        return reduce(lambda d, f: f(d), ops, ds)

    return run
