"""The vortex-shedding cross-check on a load record's time axis.

The check exists because a mis-scaled time axis is invisible to everything else:
shapes, units and ranges all stay plausible. Strouhal similarity is the one
statement about the record that does not depend on the case configuration, so it
is what catches it.
"""

from __future__ import annotations

import numpy as np
import pytest

from cfdmod.adapters.memory import MemoryFieldStore
from cfdmod.core import ElementMeta, PointsDataSource, TimeAxis, Topology
from cfdmod.dynamics import (
    STROUHAL_RECTANGULAR,
    check_vortex_shedding,
    implied_strouhal,
    spectral_peak,
    vortex_shedding_frequency,
)

N_FLOORS = 4

# The real case this came from: 86 m tower, 36.9 m across-wind face, 27.86 m/s.
U_H = 27.859
WIDTH = 36.9


def _load(series: np.ndarray, dt: float) -> PointsDataSource:
    """Spread a global across-wind series over floors as a load source."""
    per_floor = np.outer(np.linspace(0.5, 1.5, N_FLOORS), series)
    per_floor /= N_FLOORS  # so the floor sum reproduces `series`
    pts = np.column_stack(
        [np.zeros(N_FLOORS), np.zeros(N_FLOORS), np.linspace(20.0, 80.0, N_FLOORS)]
    )
    fields = {"cf_x": per_floor * 0.3, "cf_y": per_floor, "cm_z": per_floor * 0.1}
    return PointsDataSource(
        time=TimeAxis(initial_time=0.0, timestep_size=dt, n_timesteps=series.size),
        topology=Topology.points(pts),
        elements=ElementMeta(position=pts),
        fields=MemoryFieldStore(fields),
    )


def _shedding_like(f_peak: float, dt: float, n_t: int, seed: int = 0) -> np.ndarray:
    """Narrow-band across-wind force: a tone at ``f_peak`` over broadband noise."""
    rng = np.random.default_rng(seed)
    t = np.arange(n_t) * dt
    background = rng.standard_normal(n_t).cumsum() * 0.002
    return np.sin(2 * np.pi * f_peak * t) + background


@pytest.mark.unit
def test_shedding_frequency_formula():
    assert vortex_shedding_frequency(U_H, WIDTH) == pytest.approx(
        STROUHAL_RECTANGULAR * U_H / WIDTH
    )
    assert vortex_shedding_frequency(30.0, 20.0, strouhal=0.15) == pytest.approx(0.225)


@pytest.mark.unit
def test_implied_strouhal_is_the_inverse():
    f = vortex_shedding_frequency(U_H, WIDTH, strouhal=0.12)
    assert implied_strouhal(f, U_H, WIDTH) == pytest.approx(0.12)


@pytest.mark.unit
@pytest.mark.parametrize("bad", [0.0, -1.0])
def test_rejects_non_physical_inputs(bad):
    with pytest.raises(ValueError):
        vortex_shedding_frequency(bad, WIDTH)
    with pytest.raises(ValueError):
        vortex_shedding_frequency(U_H, bad)
    with pytest.raises(ValueError):
        implied_strouhal(0.1, bad, WIDTH)


@pytest.mark.unit
def test_spectral_peak_finds_the_planted_tone():
    dt, n_t, f_peak = 0.07, 8000, 0.076
    src = _load(_shedding_like(f_peak, dt, n_t), dt)
    assert spectral_peak(src) == pytest.approx(f_peak, rel=0.1)


@pytest.mark.unit
def test_a_physically_scaled_record_passes():
    """Peak where Strouhal says it should be -> implied St back at the nominal."""
    dt, n_t = 0.0705, 7955  # the real record's physical dt and length
    src = _load(_shedding_like(vortex_shedding_frequency(U_H, WIDTH), dt, n_t), dt)

    result = check_vortex_shedding(src, u_h=U_H, across_wind_width=WIDTH)

    assert result.passed, result.summary()
    assert result.implied_strouhal == pytest.approx(STROUHAL_RECTANGULAR, rel=0.15)
    assert result.frequency_ratio == pytest.approx(1.0, rel=0.15)
    result.raise_if_failed()


@pytest.mark.unit
def test_a_time_compressed_record_is_caught():
    """The real defect: the time axis de-normalised with the wrong length.

    The record is identical; only ``dt`` is wrong, compressed by B/H = 0.264 as
    it was on the issued deliverable. Every shape, unit and range still looks
    fine - and the implied Strouhal number comes out at ~0.38, which no bluff
    rectangular section produces.
    """
    dt_physical, n_t = 0.0705, 7955
    compression = 22.7 / 85.83  # base / characteristic length
    series = _shedding_like(vortex_shedding_frequency(U_H, WIDTH), dt_physical, n_t)

    good = check_vortex_shedding(_load(series, dt_physical), u_h=U_H, across_wind_width=WIDTH)
    bad = check_vortex_shedding(
        _load(series, dt_physical * compression), u_h=U_H, across_wind_width=WIDTH
    )

    assert good.passed
    assert not bad.passed, bad.summary()
    assert bad.implied_strouhal / good.implied_strouhal == pytest.approx(1 / compression, rel=0.05)
    assert bad.implied_strouhal > 0.3
    with pytest.raises(ValueError, match="mis-scaled time axis"):
        bad.raise_if_failed()


@pytest.mark.unit
def test_search_band_is_open_by_default_and_restrictable():
    """The default search must not be centred on the expected value.

    A band centred on the prediction would let the check confirm itself, so the
    default spans everything the record resolves. Narrowing it is opt-in.
    """
    dt, n_t = 0.07, 8000
    src = _load(_shedding_like(0.30, dt, n_t), dt)  # peak far from St*U/D = 0.0755

    wide = check_vortex_shedding(src, u_h=U_H, across_wind_width=WIDTH)
    assert not wide.passed
    assert wide.observed_hz == pytest.approx(0.30, rel=0.1)

    narrow = spectral_peak(src, band=(0.05, 0.12))
    assert 0.05 <= narrow <= 0.12


@pytest.mark.unit
def test_zero_variance_series_is_rejected():
    dt = 0.07
    src = _load(np.ones(2000), dt)
    with pytest.raises(ValueError, match="zero variance"):
        spectral_peak(src)


# --- Directional sweep -------------------------------------------------------

RECT = np.array([[-10.735, -18.45], [10.735, -18.45], [10.735, 18.45], [-10.735, 18.45]])
A_X, B_Y = 21.47, 36.9


def _directional_load(
    deg: float, f_peak: float, dt: float, n_t: int, *, dt_axis: float | None = None
) -> PointsDataSource:
    """Load source whose tone sits on the across-wind axis of `deg`.

    ``dt_axis`` re-labels the time axis without touching the samples - the shape
    of a mis-scaled record, where the physics is right and only the seconds
    attached to it are wrong.
    """
    t = np.arange(n_t) * dt
    a = np.deg2rad(deg)
    across = np.sin(2 * np.pi * f_peak * t)
    rng = np.random.default_rng(int(deg))
    series_x = -across * np.sin(a) + rng.standard_normal(n_t).cumsum() * 0.001
    series_y = across * np.cos(a) + rng.standard_normal(n_t).cumsum() * 0.001
    pts = np.column_stack([np.zeros(N_FLOORS), np.zeros(N_FLOORS), np.linspace(20, 80, N_FLOORS)])
    return PointsDataSource(
        time=TimeAxis(
            initial_time=0.0, timestep_size=dt if dt_axis is None else dt_axis, n_timesteps=n_t
        ),
        topology=Topology.points(pts),
        elements=ElementMeta(position=pts),
        fields=MemoryFieldStore(
            {
                "cf_x": np.tile(series_x / N_FLOORS, (N_FLOORS, 1)),
                "cf_y": np.tile(series_y / N_FLOORS, (N_FLOORS, 1)),
                "cm_z": np.zeros((N_FLOORS, n_t)),
            }
        ),
    )


@pytest.mark.unit
def test_wind_vector_convention():
    """0 deg blows along +x, 90 deg along +y - the frame cf_x / cf_y live in."""
    from cfdmod.dynamics import wind_unit_vector

    np.testing.assert_allclose(wind_unit_vector(0), [1.0, 0.0], atol=1e-12)
    np.testing.assert_allclose(wind_unit_vector(90), [0.0, 1.0], atol=1e-12)


@pytest.mark.unit
@pytest.mark.parametrize("deg", [0, 45, 90, 135, 180, 270])
def test_across_wind_width_matches_the_rectangle_formula(deg):
    """For a rectangle the projection reduces to a*|sin| + b*|cos|."""
    from cfdmod.dynamics import across_wind_width

    a = np.deg2rad(deg)
    expected = A_X * abs(np.sin(a)) + B_Y * abs(np.cos(a))
    assert across_wind_width(RECT, deg) == pytest.approx(expected, rel=1e-9)


@pytest.mark.unit
def test_across_wind_width_uses_the_wide_face_head_on():
    from cfdmod.dynamics import across_wind_width

    assert across_wind_width(RECT, 0) == pytest.approx(B_Y)
    assert across_wind_width(RECT, 90) == pytest.approx(A_X)


@pytest.mark.unit
def test_across_wind_width_rejects_bad_footprints():
    from cfdmod.dynamics import across_wind_width

    with pytest.raises(ValueError, match=r"\(n, 2\)"):
        across_wind_width(np.zeros((4, 3)), 0)
    with pytest.raises(ValueError, match="empty"):
        across_wind_width(np.zeros((0, 2)), 0)


@pytest.mark.unit
def test_across_wind_series_rotates_into_the_wind_frame():
    """At 90 deg the across-wind component is -Fx, not Fy."""
    from cfdmod.dynamics import across_wind_series

    dt, n_t = 0.07, 512
    fx = np.tile(np.linspace(1.0, 2.0, n_t), (N_FLOORS, 1))
    fy = np.tile(np.linspace(-3.0, 3.0, n_t), (N_FLOORS, 1))
    pts = np.column_stack([np.zeros(N_FLOORS), np.zeros(N_FLOORS), np.arange(N_FLOORS) * 10.0])
    src = PointsDataSource(
        time=TimeAxis(initial_time=0.0, timestep_size=dt, n_timesteps=n_t),
        topology=Topology.points(pts),
        elements=ElementMeta(position=pts),
        fields=MemoryFieldStore({"cf_x": fx, "cf_y": fy, "cm_z": fx * 0}),
    )

    np.testing.assert_allclose(across_wind_series(src, 0), fy.sum(axis=0), atol=1e-9)
    np.testing.assert_allclose(across_wind_series(src, 90), -fx.sum(axis=0), atol=1e-9)


@pytest.mark.unit
def test_strouhal_sweep_is_flat_for_a_dimensionally_consistent_case():
    """A record that sheds at St*U/D in every direction reads back that St."""
    from cfdmod.dynamics import across_wind_width, strouhal_by_direction

    dt, n_t, st_true = 0.0705, 7955, 0.10
    loads = {
        float(deg): _directional_load(
            deg,
            vortex_shedding_frequency(U_H, across_wind_width(RECT, deg), strouhal=st_true),
            dt,
            n_t,
        )
        for deg in range(0, 360, 45)
    }

    table = strouhal_by_direction(loads, RECT, u_h_by_direction=U_H)

    assert list(table["direction_deg"]) == sorted(table["direction_deg"])
    assert table["passed"].all(), table
    np.testing.assert_allclose(table["implied_strouhal"], st_true, rtol=0.12)
    # the width really does vary with direction, so this was not trivially flat
    assert table["across_wind_width_m"].max() / table["across_wind_width_m"].min() > 1.5


@pytest.mark.unit
def test_strouhal_sweep_flags_a_globally_compressed_record():
    """A wrong time axis displaces the whole curve, not one point."""
    from cfdmod.dynamics import across_wind_width, strouhal_by_direction

    dt, n_t, st_true = 0.0705, 7955, 0.10
    compression = 22.7 / 85.83
    loads = {}
    for deg in range(0, 360, 45):
        f = vortex_shedding_frequency(U_H, across_wind_width(RECT, deg), strouhal=st_true)
        # identical samples, only the seconds attached to them are wrong
        loads[float(deg)] = _directional_load(deg, f, dt, n_t, dt_axis=dt * compression)

    table = strouhal_by_direction(loads, RECT, u_h_by_direction=U_H)

    assert not table["passed"].any(), table
    np.testing.assert_allclose(table["implied_strouhal"], st_true / compression, rtol=0.12)


@pytest.mark.unit
def test_plot_strouhal_by_direction_marks_the_failures():
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import pandas as pd

    from cfdmod.dynamics import plot_strouhal_by_direction

    table = pd.DataFrame(
        {
            "direction_deg": [0.0, 90.0, 180.0],
            "implied_strouhal": [0.10, 0.38, 0.11],
            "passed": [True, False, True],
        }
    )
    fig, ax = plot_strouhal_by_direction(table, language="pt-br")
    labels = [t.get_text() for t in ax.get_legend().get_texts()]
    assert any("fora da faixa" in lab for lab in labels), labels
    plt.close(fig)
