"""Unit tests for the extreme-value / peak op.

Covers the peak-factor identity, a Gumbel known-answer on a synthetic
draw, event-duration rescaling monotonicity, the time-collapse contract,
and parameter validation.
"""

from __future__ import annotations

import numpy as np
import pytest

from cfdmod.adapters.memory import MemoryFieldStore
from cfdmod.core import ElementMeta, SurfaceDataSource, TimeAxis, Topology
from cfdmod.core.ops.data_source_create.extreme_value import (
    ExtremeValueParams,
    extreme_value,
    gumbel_extreme_value_1d,
    reescale_event_duration_peak,
)


def _surface(values: np.ndarray, dt: float = 0.01) -> SurfaceDataSource:
    n_elements, n_timesteps = values.shape
    verts = np.tile(np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0]]), (n_elements, 1)).astype(
        np.float64
    )
    tris = (np.arange(n_elements * 3).reshape(n_elements, 3)).astype(np.int32)
    return SurfaceDataSource(
        time=TimeAxis(initial_time=0.0, timestep_size=dt, n_timesteps=n_timesteps),
        topology=Topology.triangles(tris, verts),
        elements=ElementMeta(),
        fields=MemoryFieldStore({"pressure": values.astype(np.float64)}),
    )


# --- peak factor -----------------------------------------------------------


def test_peak_factor_max_identity():
    rng = np.random.default_rng(0)
    data = rng.normal(size=(4, 500))
    ds = _surface(data)
    k = 3.5
    out = extreme_value(
        ds, ExtremeValueParams(method="peak_factor", extreme_type="max", peak_factor=k)
    )
    mean = data.mean(axis=1)
    rms = (data - mean[:, None]).std(axis=1)
    np.testing.assert_allclose(out.fields.read("peak_factor_max"), mean + k * rms, rtol=1e-12)


def test_peak_factor_min_identity():
    rng = np.random.default_rng(1)
    data = rng.normal(size=(3, 400))
    ds = _surface(data)
    k = 2.0
    out = extreme_value(
        ds, ExtremeValueParams(method="peak_factor", extreme_type="min", peak_factor=k)
    )
    mean = data.mean(axis=1)
    rms = (data - mean[:, None]).std(axis=1)
    np.testing.assert_allclose(out.fields.read("peak_factor_min"), mean - k * rms, rtol=1e-12)


def test_output_collapses_time_axis():
    data = np.random.default_rng(2).normal(size=(5, 300))
    ds = _surface(data)
    out = extreme_value(
        ds, ExtremeValueParams(method="peak_factor", extreme_type="max", peak_factor=3.0)
    )
    assert out.time.is_time_aggregated
    assert out.fields.read("peak_factor_max").shape == (5,)
    assert out.field_names == ["peak_factor_max"]


def test_custom_out_name():
    data = np.random.default_rng(3).normal(size=(2, 200))
    ds = _surface(data)
    out = extreme_value(
        ds, ExtremeValueParams(method="peak_factor", extreme_type="max", peak_factor=3.0, out="ce")
    )
    assert out.field_names == ["ce"]


# --- gumbel ----------------------------------------------------------------


def test_gumbel_known_answer_on_synthetic_draw():
    # peak_duration below dt -> window of 1 sample -> smoothing is identity,
    # so block maxima are the raw sub-array maxima of a known distribution.
    dt = 1.0
    loc_true, scale_true = 5.0, 1.5
    rng = np.random.default_rng(42)
    # Large series so the block-maxima Gumbel fit is well conditioned.
    series = rng.gumbel(loc=loc_true, scale=scale_true, size=200000)
    est = gumbel_extreme_value_1d(
        series,
        dt=dt,
        peak_duration=dt / 2,  # 1-sample window
        event_duration=dt * len(series) / 10,  # == sub-window duration -> no rescale
        extreme_type="max",
        n_subdivisions=10,
        non_exceedance_probability=0.78,
    )
    # Block-max of N gumbel samples is gumbel(loc + scale*ln N, scale).
    n_block = len(series) / 10
    exp_loc = loc_true + scale_true * np.log(n_block)
    from scipy.stats import gumbel_r

    expected = gumbel_r.ppf(0.78, loc=exp_loc, scale=scale_true)
    assert est == pytest.approx(expected, rel=0.05)


def test_gumbel_op_shape_and_ordering():
    rng = np.random.default_rng(7)
    data = rng.normal(size=(3, 5000))
    ds = _surface(data, dt=0.01)
    p = ExtremeValueParams(
        method="gumbel",
        extreme_type="max",
        peak_duration=0.03,
        event_duration=600.0,
        n_subdivisions=10,
    )
    out = extreme_value(ds, p)
    gmax = out.fields.read("gumbel_max")
    assert gmax.shape == (3,)
    # A max extreme should exceed the sample mean for each element.
    assert np.all(gmax > data.mean(axis=1))


# --- event-duration rescale ------------------------------------------------


def test_rescale_longer_event_raises_max_lowers_min():
    loc, scale = 10.0, 2.0
    loc_up, scale_up = reescale_event_duration_peak(loc, scale, 60.0, 600.0, "max")
    loc_dn, scale_dn = reescale_event_duration_peak(loc, scale, 60.0, 600.0, "min")
    assert scale_up == scale and scale_dn == scale
    assert loc_up > loc  # longer window -> larger expected max
    assert loc_dn < loc  # longer window -> smaller expected min


def test_rescale_identity_when_durations_equal():
    loc, scale = 3.0, 1.0
    out_loc, out_scale = reescale_event_duration_peak(loc, scale, 300.0, 300.0, "max")
    assert (out_loc, out_scale) == pytest.approx((loc, scale))


# --- validation ------------------------------------------------------------


def test_gumbel_requires_durations():
    with pytest.raises(ValueError):
        ExtremeValueParams(method="gumbel", extreme_type="max")


def test_peak_factor_requires_factor():
    with pytest.raises(ValueError):
        ExtremeValueParams(method="peak_factor", extreme_type="max")


def test_peak_factor_rejects_gumbel_params():
    with pytest.raises(ValueError):
        ExtremeValueParams(
            method="peak_factor", extreme_type="max", peak_factor=3.0, peak_duration=1.0
        )


def test_peak_factor_rejects_n_subdivisions():
    with pytest.raises(ValueError, match="peak_factor"):
        ExtremeValueParams(
            method="peak_factor", extreme_type="max", peak_factor=3.0, n_subdivisions=5
        )


def test_peak_factor_rejects_non_exceedance_probability():
    with pytest.raises(ValueError, match="peak_factor"):
        ExtremeValueParams(
            method="peak_factor",
            extreme_type="max",
            peak_factor=3.0,
            non_exceedance_probability=0.9,
        )


def test_gumbel_peak_duration_too_long_raises():
    # peak_duration larger than the record shrinks the smoothed series
    # below n_subdivisions; expect a clear error, not np.max on empty.
    dt = 0.1
    data = np.random.default_rng(0).normal(size=(1, 20))
    ds = _surface(data, dt=dt)
    p = ExtremeValueParams(
        method="gumbel",
        extreme_type="max",
        peak_duration=5.0,  # 50 samples > 20 -> empty valid-mode convolution
        event_duration=600.0,
    )
    with pytest.raises(ValueError, match="exceeds|n_subdivisions|shorter"):
        extreme_value(ds, p)


def test_gumbel_smoothed_shorter_than_subdivisions_raises():
    # window <= N but the smoothed series is still shorter than
    # n_subdivisions -> caught before np.max hits an empty block.
    dt = 0.1
    data = np.random.default_rng(1).normal(size=(1, 20))
    ds = _surface(data, dt=dt)
    p = ExtremeValueParams(
        method="gumbel",
        extreme_type="max",
        peak_duration=1.6,  # 16 samples -> smoothed size 5 < 10 subdivisions
        event_duration=600.0,
    )
    with pytest.raises(ValueError, match="shorter than n_subdivisions"):
        extreme_value(ds, p)


def test_gumbel_defaults_applied_when_none():
    rng = np.random.default_rng(11)
    data = rng.normal(size=(1, 5000))
    ds = _surface(data, dt=0.01)
    # n_subdivisions / non_exceedance_probability left unset (None).
    p = ExtremeValueParams(
        method="gumbel", extreme_type="max", peak_duration=0.03, event_duration=600.0
    )
    out = extreme_value(ds, p)
    assert out.fields.read("gumbel_max").shape == (1,)


def test_gumbel_rejects_peak_factor():
    with pytest.raises(ValueError):
        ExtremeValueParams(
            method="gumbel",
            extreme_type="max",
            peak_duration=1.0,
            event_duration=10.0,
            peak_factor=3.0,
        )


def test_rejects_time_aggregated_source():
    verts = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0]], dtype=np.float64)
    tris = np.array([[0, 1, 2]], dtype=np.int32)
    ds = SurfaceDataSource(
        time=TimeAxis(initial_time=0.0, timestep_size=0.0, n_timesteps=0),
        topology=Topology.triangles(tris, verts),
        elements=ElementMeta(),
        fields=MemoryFieldStore({"pressure": np.array([1.0])}),
    )
    with pytest.raises(ValueError):
        extreme_value(
            ds, ExtremeValueParams(method="peak_factor", extreme_type="max", peak_factor=3.0)
        )
