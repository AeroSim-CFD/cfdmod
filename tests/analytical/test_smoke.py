"""Smoke tests for cfdmod.analytical.

These pin the public API shape and basic NBR/EU wind-profile behaviour
so that future numpy bumps or accidental refactors don't drift the
module silently.
"""

from __future__ import annotations

import pandas as pd
import pytest

from cfdmod.analytical import WindProfile_EU, WindProfile_NBR

pytestmark = pytest.mark.unit


def _nbr_directional_data() -> pd.DataFrame:
    """Minimal NBR directional table with a single direction in category II."""
    return pd.DataFrame(
        {
            "wind_direction": [0.0, 90.0, 180.0, 270.0],
            "I": [0.0, 0.0, 0.0, 0.0],
            "II": [1.0, 1.0, 1.0, 1.0],
            "III": [0.0, 0.0, 0.0, 0.0],
            "IV": [0.0, 0.0, 0.0, 0.0],
            "V": [0.0, 0.0, 0.0, 0.0],
            "Kd": [1.0, 1.0, 1.0, 1.0],
        }
    )


def test_nbr_construct_and_S2_increases_with_height():
    wp = WindProfile_NBR(directional_data=_nbr_directional_data(), V0=35.0)
    s2_low = wp.S2(height=10.0, direction=0.0, time_filter_seconds=600)
    s2_high = wp.S2(height=100.0, direction=0.0, time_filter_seconds=600)
    assert s2_low > 0
    assert s2_high > s2_low


def test_nbr_S3_recurrence_monotone():
    """S3 should grow with recurrence period (longer return = higher factor)."""
    wp = WindProfile_NBR(directional_data=_nbr_directional_data(), V0=35.0)
    s3_50 = wp.S3(50)
    s3_100 = wp.S3(100)
    s3_500 = wp.S3(500)
    assert s3_50 < s3_100 < s3_500


def test_nbr_get_U_H_overwrite_short_circuits():
    wp = WindProfile_NBR(
        directional_data=_nbr_directional_data(), V0=35.0, U_H_overwrite=42.0
    )
    u_h = wp.get_U_H(height=10.0, direction=0.0, recurrence_period=50)
    assert u_h == 42.0


def test_nbr_get_U_H_scales_with_V0():
    """Doubling V0 should double the resulting U_H (linear in V0)."""
    df = _nbr_directional_data()
    wp1 = WindProfile_NBR(directional_data=df.copy(), V0=20.0)
    wp2 = WindProfile_NBR(directional_data=df.copy(), V0=40.0)
    u1 = wp1.get_U_H(height=10.0, direction=0.0, recurrence_period=50)
    u2 = wp2.get_U_H(height=10.0, direction=0.0, recurrence_period=50)
    assert u2 == pytest.approx(2.0 * u1, rel=1e-6)


def test_nbr_get_opencountry_profile_returns_base_class():
    wp = WindProfile_NBR(directional_data=_nbr_directional_data(), V0=35.0)
    out = wp.get_opencountry_profile()
    # Should be a WindProfile-like object with directional_data still present.
    assert hasattr(out, "directional_data")
    # The "open country" profile is forced to category II only.
    df = out.directional_data
    assert (df["II"] == 1).all()
    for cat in ["I", "III", "IV", "V"]:
        assert (df[cat] == 0).all()


def test_eu_class_is_importable():
    """Smoke: just verify WindProfile_EU is the class exported by the
    public API. (Construction signature differs from NBR; not exercised
    here, but pinning the import keeps the public surface honest.)"""
    assert WindProfile_EU.__name__ == "WindProfile_EU"
