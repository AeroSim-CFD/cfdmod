"""Unit tests for the Butterworth frequency-filter field op.

Passband / stopband behaviour is checked on a two-tone signal; the
zero-phase default is checked against a causal single pass; parameter
validation (cutoff arity, Nyquist bound) is exercised.
"""

from __future__ import annotations

import numpy as np
import pytest

from cfdmod.adapters.memory import MemoryFieldStore
from cfdmod.core import ElementMeta, SurfaceDataSource, TimeAxis, Topology
from cfdmod.core.ops.field.frequency_filter import FrequencyFilterParams, frequency_filter


def _surface(values: np.ndarray, dt: float) -> SurfaceDataSource:
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


def _two_tone(dt: float, n: int, f_lo: float, f_hi: float):
    t = np.arange(n) * dt
    slow = np.sin(2 * np.pi * f_lo * t)
    fast = np.sin(2 * np.pi * f_hi * t)
    return t, slow, fast, (slow + fast).reshape(1, -1)


def test_lowpass_keeps_slow_rejects_fast():
    dt = 1e-3
    _, slow, fast, sig = _two_tone(dt, 4000, f_lo=2.0, f_hi=80.0)
    ds = _surface(sig, dt)
    out = frequency_filter(ds, FrequencyFilterParams(btype="lowpass", cutoff=10.0))
    y = out.fields.read("pressure")[0]
    # slow tone preserved, fast tone strongly attenuated.
    assert np.std(y - slow) < 0.1 * np.std(slow)
    assert np.std(y) < 1.2 * np.std(slow)


def test_highpass_keeps_fast_rejects_slow():
    dt = 1e-3
    _, slow, fast, sig = _two_tone(dt, 4000, f_lo=2.0, f_hi=80.0)
    ds = _surface(sig, dt)
    out = frequency_filter(ds, FrequencyFilterParams(btype="highpass", cutoff=30.0))
    y = out.fields.read("pressure")[0]
    assert np.std(y - fast) < 0.1 * np.std(fast)


def test_bandpass_isolates_mid_tone():
    dt = 1e-3
    t = np.arange(4000) * dt
    lo = np.sin(2 * np.pi * 2.0 * t)
    mid = np.sin(2 * np.pi * 40.0 * t)
    hi = np.sin(2 * np.pi * 200.0 * t)
    ds = _surface((lo + mid + hi).reshape(1, -1), dt)
    out = frequency_filter(ds, FrequencyFilterParams(btype="bandpass", cutoff=(20.0, 60.0)))
    y = out.fields.read("pressure")[0]
    assert np.std(y - mid) < 0.15 * np.std(mid)


def test_bandstop_rejects_mid_tone():
    dt = 1e-3
    t = np.arange(4000) * dt
    lo = np.sin(2 * np.pi * 2.0 * t)
    mid = np.sin(2 * np.pi * 40.0 * t)
    ds = _surface((lo + mid).reshape(1, -1), dt)
    out = frequency_filter(ds, FrequencyFilterParams(btype="bandstop", cutoff=(20.0, 60.0)))
    y = out.fields.read("pressure")[0]
    assert np.std(y - lo) < 0.15 * np.std(lo)


def test_zero_phase_has_no_lag_causal_does():
    dt = 1e-3
    t = np.arange(3000) * dt
    # single low tone plus a burst of noise to filter out.
    rng = np.random.default_rng(0)
    sig = (np.sin(2 * np.pi * 3.0 * t) + 0.5 * rng.normal(size=t.size)).reshape(1, -1)
    ds = _surface(sig, dt)
    zp = frequency_filter(ds, FrequencyFilterParams(btype="lowpass", cutoff=8.0, zero_phase=True))
    causal = frequency_filter(
        ds, FrequencyFilterParams(btype="lowpass", cutoff=8.0, zero_phase=False)
    )
    clean = np.sin(2 * np.pi * 3.0 * t)
    # zero-phase tracks the clean tone better (no group delay).
    err_zp = np.std(zp.fields.read("pressure")[0][100:-100] - clean[100:-100])
    err_causal = np.std(causal.fields.read("pressure")[0][100:-100] - clean[100:-100])
    assert err_zp < err_causal


def test_out_field_leaves_source_intact():
    dt = 1e-3
    _, _, _, sig = _two_tone(dt, 2000, 2.0, 80.0)
    ds = _surface(sig, dt)
    out = frequency_filter(
        ds, FrequencyFilterParams(btype="lowpass", cutoff=10.0, out="pressure_lp")
    )
    assert sorted(out.field_names) == ["pressure", "pressure_lp"]
    np.testing.assert_array_equal(out.fields.read("pressure"), sig)


def test_scalar_cutoff_rejected_for_band():
    with pytest.raises(ValueError):
        FrequencyFilterParams(btype="bandpass", cutoff=10.0)


def test_pair_cutoff_rejected_for_lowpass():
    with pytest.raises(ValueError):
        FrequencyFilterParams(btype="lowpass", cutoff=(10.0, 20.0))


def test_unordered_band_cutoff_rejected():
    with pytest.raises(ValueError):
        FrequencyFilterParams(btype="bandpass", cutoff=(60.0, 20.0))


def test_cutoff_above_nyquist_rejected():
    dt = 1e-2  # fs = 100 Hz, Nyquist = 50 Hz
    data = np.zeros((1, 100))
    ds = _surface(data, dt)
    with pytest.raises(ValueError):
        frequency_filter(ds, FrequencyFilterParams(btype="lowpass", cutoff=60.0))


def test_zero_phase_short_series_raises_actionable_error():
    # 10 samples passes the n_timesteps>=2 guard but is below the
    # order-4 sosfiltfilt padlen; expect a clear cfdmod error, not a
    # cryptic scipy one.
    dt = 1e-3
    data = np.random.default_rng(0).normal(size=(1, 10))
    ds = _surface(data, dt)
    with pytest.raises(ValueError, match="longer record|lower order|zero_phase"):
        frequency_filter(ds, FrequencyFilterParams(btype="lowpass", cutoff=50.0))
