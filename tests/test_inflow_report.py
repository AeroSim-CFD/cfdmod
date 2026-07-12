"""Tests for the code-standard comparison helpers in cfdmod.inflow_report."""

from __future__ import annotations

import matplotlib

matplotlib.use("Agg")  # headless

import pathlib  # noqa: E402

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import pytest  # noqa: E402

from cfdmod import inflow_report as ir  # noqa: E402
from cfdmod.analytical import WindProfile_NBR  # noqa: E402

pytestmark = pytest.mark.unit

REPO = pathlib.Path(__file__).resolve().parents[1]
INFLOW_FIX = REPO / "fixtures" / "tests" / "inflow" / "pitot_inlet"
WIND_FIX = REPO / "fixtures" / "tests" / "inflow" / "wind_analysis"


def _nbr_profile() -> WindProfile_NBR:
    df = pd.DataFrame(
        {
            "wind_direction": [0.0, 90.0, 180.0, 270.0],
            "I": [0.0] * 4,
            "II": [1.0] * 4,
            "III": [0.0] * 4,
            "IV": [0.0] * 4,
            "V": [0.0] * 4,
            "Kd": [1.0, 1.2, 1.0, 1.0],
        }
    )
    return WindProfile_NBR(directional_data=df, V0=35.0)


def test_directional_reference_speed_all_directions():
    s = ir.directional_reference_speed(_nbr_profile(), height=100.0, recurrence_period=50)
    assert list(s.index) == [0.0, 90.0, 180.0, 270.0]
    assert (s.to_numpy() > 0).all()
    assert s.max() == pytest.approx(float(s.to_numpy().max()))


def test_directional_reference_speed_subset_and_kd():
    wp = _nbr_profile()
    base = ir.directional_reference_speed(wp, height=50.0, directions=[90.0], use_kd=False)
    with_kd = ir.directional_reference_speed(wp, height=50.0, directions=[90.0], use_kd=True)
    # direction 90 has Kd=1.2, so enabling Kd scales U_H up by 1.2
    assert with_kd.loc[90.0] == pytest.approx(1.2 * base.loc[90.0], rel=1e-6)


def test_directional_reference_speed_from_fixture_csvs():
    from cfdmod.analytical import WindProfile_EU, WindProfile_NBR

    nbr = WindProfile_NBR.build(WIND_FIX / "wind_analysis_NBR.csv", V0=35.0)
    eu = WindProfile_EU.build(WIND_FIX / "wind_analysis_EU.csv", Vb=35.0)
    s_nbr = ir.directional_reference_speed(nbr, height=100.0, recurrence_period=50, use_kd=True)
    s_eu = ir.directional_reference_speed(eu, height=100.0, recurrence_period=50, use_kd=True)
    assert len(s_nbr) == 16 and len(s_eu) == 16
    assert (s_nbr.to_numpy() > 0).all() and (s_eu.to_numpy() > 0).all()


def test_eu_integral_length_scale_monotone_and_positive():
    z = np.array([10.0, 50.0, 100.0, 200.0, 300.0])
    L = ir.eu_integral_length_scale(z, z0=0.3)
    assert np.all(L > 0)
    assert np.all(np.diff(L) > 0)
    assert L[np.argmin(np.abs(z - 200.0))] == pytest.approx(300.0, rel=1e-6)  # L(z_t)=L_t


def test_plot_integral_length_scale_returns_fig_ax():
    z = np.linspace(1.0, 200.0, 20)
    L = ir.eu_integral_length_scale(z, z0=0.3)
    fig, ax = ir.plot_integral_length_scale(z, L, H=200.0, L_theory=L)
    assert fig is not None and ax is not None


@pytest.mark.integration
def test_profile_vs_code_on_pitot_fixture():
    if not (INFLOW_FIX / "hist_series.csv").exists():
        pytest.skip("pitot_inlet fixture not present")
    from cfdmod.inflow import InflowData

    inflow = InflowData.from_files(INFLOW_FIX / "hist_series.csv", INFLOW_FIX / "points.csv")
    profiles = ir.detect_profiles(inflow, min_points=3)
    assert profiles
    prof = profiles[0]
    ref_h = float(np.median(prof.z))

    fig, ax = ir.plot_profile_vs_code(prof, inflow, ref_h, cat_eu="III")
    assert fig is not None and len(ax) == 2

    scales = ir.integral_length_scale_profile(inflow, prof)
    assert scales.shape == prof.z.shape
