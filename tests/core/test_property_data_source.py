"""Property-based DataSource-invariant tests (issue #141 phase 2).

The frozen :class:`~cfdmod.core.data_source.DataSource` advertises that every
functional update either returns a source that still passes
``_check_consistency`` or raises -- it never silently produces a shape-
inconsistent frozen object. That exact bug (a functional update bypassing the
validators) is one of the regressions pinned in
``tests/core/test_review_regressions.py``; here we assert it generatively.

Note: pydantic v2 wraps the ``ValueError`` raised inside a model validator in a
:class:`pydantic.ValidationError` (which is *not* a ``ValueError`` subclass), so
inconsistency-should-raise assertions accept either type.
"""

from __future__ import annotations

import numpy as np
import pytest
from hypothesis import given, settings
from pydantic import ValidationError

from cfdmod.core.grouping import Grouping
from cfdmod.core.time_axis import TimeAxis
from cfdmod.core.topology import ElementMeta
from tests import strategies as sty

pytestmark = pytest.mark.property

_INCONSISTENT = (ValueError, ValidationError)


def _revalidates(ds) -> bool:
    """True if ``ds`` still satisfies every model validator."""
    type(ds).model_validate(dict(ds.__dict__))
    return True


# ---------------------------------------------------------------------------
# Positive: metadata-only updates stay consistent and preserve n_elements.
# ---------------------------------------------------------------------------


@settings(max_examples=50)
@given(ds=sty.data_sources())
def test_metadata_updates_preserve_n_elements(ds) -> None:
    n = ds.n_elements
    # with_attrs and without_grouping take the plain-model_copy path (no
    # re-validation), so they are the most likely to drift; assert they stay
    # consistent and n_elements is stable.
    with_attrs = ds.with_attrs(note="x")
    assert with_attrs.n_elements == n
    assert _revalidates(with_attrs)

    dropped = ds.without_grouping("does-not-exist")
    assert dropped.n_elements == n
    assert _revalidates(dropped)


@settings(max_examples=50)
@given(ds=sty.data_sources())
def test_compatible_with_time_preserves_shape(ds) -> None:
    # A translate keeps n_timesteps, so field shapes still match -> consistent.
    new_time = ds.time.translate(ds.time.initial_time + 5.0)
    updated = ds.with_time(new_time)
    assert updated.n_elements == ds.n_elements
    assert updated.time.n_timesteps == ds.time.n_timesteps
    assert _revalidates(updated)


@settings(max_examples=50)
@given(ds=sty.data_sources())
def test_valid_with_field_stays_consistent(ds) -> None:
    n = ds.n_elements
    shape = (n,) if ds.time.is_time_aggregated else (n, ds.time.n_timesteps)
    updated = ds.with_field("extra", np.ones(shape, dtype=np.float64))
    assert updated.n_elements == n
    assert "extra" in updated.field_names
    assert _revalidates(updated)


# ---------------------------------------------------------------------------
# Negative: an inconsistent update must raise, never silently succeed.
# ---------------------------------------------------------------------------


@settings(max_examples=50)
@given(ds=sty.data_sources())
def test_with_field_wrong_leading_axis_raises(ds) -> None:
    bad = np.zeros((ds.n_elements + 1,), dtype=np.float64)
    with pytest.raises(_INCONSISTENT):
        ds.with_field("bad", bad)


@settings(max_examples=50)
@given(ds=sty.data_sources())
def test_with_grouping_wrong_size_raises(ds) -> None:
    bad = Grouping(name="bad", indices=np.zeros(ds.n_elements + 1, dtype=np.int32))
    with pytest.raises(_INCONSISTENT):
        ds.with_grouping(bad)


@settings(max_examples=50)
@given(ds=sty.data_sources())
def test_with_elements_wrong_size_raises(ds) -> None:
    bad = ElementMeta(position=np.zeros((ds.n_elements + 1, 3), dtype=np.float64))
    with pytest.raises(_INCONSISTENT):
        ds.with_elements(bad)


@settings(max_examples=50)
@given(ds=sty.data_sources())
def test_with_time_incompatible_axis_raises(ds) -> None:
    if ds.time.is_time_aggregated:
        # Aggregated source: fields are 1-D, so attaching a real time axis
        # makes the field time-length (0) disagree with n_timesteps.
        bad_time = TimeAxis(initial_time=0.0, timestep_size=1.0, n_timesteps=2)
    else:
        # Time-resolved: change the length so field time axes no longer match.
        bad_time = TimeAxis(
            initial_time=ds.time.initial_time,
            timestep_size=ds.time.timestep_size,
            n_timesteps=ds.time.n_timesteps + 1,
        )
    with pytest.raises(_INCONSISTENT):
        ds.with_time(bad_time)
