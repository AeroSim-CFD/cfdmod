"""Property-based algebra tests (issue #141 phase 2).

Exercises the four broadcasting rules in :mod:`cfdmod.core.algebra` over many
random inputs. Float caveats baked in:

- Commutativity is asserted only for the ``elementwise`` and ``constant`` rules.
  ``classify_broadcast`` is asymmetric -- ``column`` needs the rhs to be the
  single-element operand and ``row`` needs exactly one side time-aggregated --
  so ``a op b`` and ``b op a`` do not generally classify the same way.
- ``a - a`` is exactly zero for finite floats; ``(a * k) / k == a`` is only
  recovered up to a floating-point tolerance.
"""

from __future__ import annotations

import numpy as np
import pytest
from hypothesis import given
from hypothesis import strategies as st
from hypothesis.extra import numpy as hnp

from cfdmod.adapters.memory import MemoryFieldStore
from cfdmod.core import algebra
from cfdmod.core.data_source import PointsDataSource
from cfdmod.core.time_axis import TimeAxis
from cfdmod.core.topology import ElementMeta, Topology

pytestmark = pytest.mark.property

FIELD = "v"


def _points(arr: np.ndarray) -> PointsDataSource:
    """A points source carrying ``arr`` under field ``v``.

    A 1-D array is treated as a time-aggregated source; a 2-D array as a
    time-resolved one with ``n_timesteps == arr.shape[1]``.
    """
    arr = np.asarray(arr, dtype=np.float64)
    n = arr.shape[0]
    if arr.ndim == 1:
        time = TimeAxis(initial_time=0.0, timestep_size=0.0, n_timesteps=0)
    else:
        time = TimeAxis(initial_time=0.0, timestep_size=1.0, n_timesteps=arr.shape[1])
    return PointsDataSource(
        time=time,
        topology=Topology.points(np.zeros((n, 3), dtype=np.float64)),
        elements=ElementMeta(),
        fields=MemoryFieldStore({FIELD: arr}),
    )


def _finite(**kw):
    return st.floats(allow_nan=False, allow_infinity=False, width=64, **kw)


@st.composite
def _arrays(draw, *, allow_1d: bool = True):
    """A finite float64 array, 1-D (aggregated) or 2-D (time-resolved)."""
    n = draw(st.integers(min_value=1, max_value=6))
    ndim = draw(st.sampled_from([1, 2])) if allow_1d else 2
    shape = (n,) if ndim == 1 else (n, draw(st.integers(min_value=1, max_value=5)))
    return draw(hnp.arrays(np.float64, shape, elements=_finite(min_value=-1e6, max_value=1e6)))


@given(arr=_arrays())
def test_sub_self_is_zero(arr: np.ndarray) -> None:
    a = _points(arr)
    out = algebra.sub(a, a, field=FIELD)
    assert np.array_equal(out.fields.read(FIELD), np.zeros_like(arr))


@given(
    arr=_arrays(),
    k=st.one_of(_finite(min_value=1e-3, max_value=1e3), _finite(min_value=-1e3, max_value=-1e-3)),
)
def test_mul_then_div_recovers_original(arr: np.ndarray, k: float) -> None:
    a = _points(arr)
    scaled = algebra.mul(a, k, field=FIELD)
    back = algebra.div(scaled, k, field=FIELD)
    assert np.allclose(back.fields.read(FIELD), arr, rtol=1e-9, atol=1e-9)


@given(data=st.data())
def test_add_and_mul_commute_elementwise(data) -> None:
    arr1 = data.draw(_arrays())
    # Second operand shares the exact shape so the rule is elementwise.
    arr2 = data.draw(
        hnp.arrays(np.float64, arr1.shape, elements=_finite(min_value=-1e6, max_value=1e6))
    )
    a, b = _points(arr1), _points(arr2)
    assert np.array_equal(
        algebra.add(a, b, field=FIELD).fields.read(FIELD),
        algebra.add(b, a, field=FIELD).fields.read(FIELD),
    )
    assert np.array_equal(
        algebra.mul(a, b, field=FIELD).fields.read(FIELD),
        algebra.mul(b, a, field=FIELD).fields.read(FIELD),
    )


@given(
    n=st.integers(min_value=1, max_value=6),
    nt=st.integers(min_value=1, max_value=5),
    rule=st.sampled_from(["constant", "elementwise", "column", "row"]),
)
def test_broadcast_result_shape_matches_lhs(n: int, nt: int, rule: str) -> None:
    lhs = _points(np.ones((n, nt), dtype=np.float64))
    if rule == "constant":
        out = algebra.mul(lhs, 2.0, field=FIELD)
    elif rule == "elementwise":
        out = algebra.mul(lhs, _points(np.full((n, nt), 3.0)), field=FIELD)
    elif rule == "column":
        # rhs single element, same time axis -> broadcast across elements.
        out = algebra.mul(lhs, _points(np.full((1, nt), 3.0)), field=FIELD)
    else:  # row: rhs time-aggregated (n,) -> broadcast across the time axis.
        out = algebra.mul(lhs, _points(np.full((n,), 3.0)), field=FIELD)
    assert out.fields.read(FIELD).shape == lhs.fields.read(FIELD).shape == (n, nt)
