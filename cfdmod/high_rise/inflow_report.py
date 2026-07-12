"""Inflow (ABL) validation: detect vertical profiles and build the figures.

Wraps the pure functions in :mod:`cfdmod.inflow`. The old notebooks read the
probe-line layout from the solver config; here we *auto-detect* vertical
profiles from the probe point cloud (points sharing an (x, y) column, varying
z), so a case does not need to hand-list its lines.

Figures produced per profile:
    - mean streamwise velocity vs height
    - turbulence intensity vs height
    - normalized velocity spectrum at the reference height
and a scalar integral length scale estimate. The high-rise sequence also uses
:func:`reference_velocity` to read U_H off the mean profile at the interest
height.
"""

from __future__ import annotations

import dataclasses

import numpy as np
import pandas as pd
from scipy.optimize import curve_fit

from cfdmod.inflow import (
    InflowData,
    NormalizationParameters,
    calculate_mean_velocity,
    calculate_spectral_density,
    calculate_turbulence_intensity,
)

from .plotting import new_axes


@dataclasses.dataclass(frozen=True)
class ProfileLine:
    """A detected vertical profile: point indices ordered by ascending height."""

    name: str
    x: float
    y: float
    point_idx: np.ndarray  # ordered by z
    z: np.ndarray  # ascending heights

    def nearest_index(self, height: float) -> int:
        """point_idx whose height is closest to ``height``."""
        return int(self.point_idx[int(np.abs(self.z - height).argmin())])


def detect_profiles(
    inflow: InflowData,
    *,
    xy_tol: float = 1e-3,
    min_points: int = 3,
) -> list[ProfileLine]:
    """Group probe points into vertical profiles by shared (x, y) column.

    Points are bucketed by (x, y) rounded to ``xy_tol``; a bucket with at
    least ``min_points`` distinct heights is a vertical profile. Profiles are
    returned sorted by descending point count (richest first).
    """
    pts = inflow.points.copy()
    idx_col = "idx" if "idx" in pts.columns else pts.index.name or "index"
    if "idx" not in pts.columns:
        pts = pts.reset_index().rename(columns={pts.index.name or "index": "idx"})
        idx_col = "idx"

    decimals = max(0, int(round(-np.log10(xy_tol))))
    pts["_kx"] = pts["x"].round(decimals)
    pts["_ky"] = pts["y"].round(decimals)

    profiles: list[ProfileLine] = []
    for (kx, ky), grp in pts.groupby(["_kx", "_ky"]):
        grp = grp.sort_values("z")
        if grp["z"].nunique() < min_points:
            continue
        profiles.append(
            ProfileLine(
                name=f"x{kx:g}_y{ky:g}",
                x=float(grp["x"].iloc[0]),
                y=float(grp["y"].iloc[0]),
                point_idx=grp[idx_col].to_numpy(),
                z=grp["z"].to_numpy(dtype=float),
            )
        )
    profiles.sort(key=lambda p: len(p.point_idx), reverse=True)
    return profiles


def _profile_series(profile: ProfileLine, per_point: pd.DataFrame, column: str) -> np.ndarray:
    """Pull ``column`` for this profile's points, ordered by height."""
    indexed = per_point.set_index("point_idx")[column]
    return indexed.loc[profile.point_idx].to_numpy(dtype=float)


def reference_velocity(
    profile: ProfileLine,
    inflow: InflowData,
    reference_height: float,
    component: str = "ux",
) -> float:
    """Simulation mean velocity interpolated onto ``reference_height``."""
    means = calculate_mean_velocity(inflow, for_components=[component])
    u = _profile_series(profile, means, f"{component}_mean")
    return float(np.interp(reference_height, profile.z, u))


def integral_length_scale(
    inflow: InflowData,
    point_idx: int,
    u_mean: float,
    component: str = "ux",
) -> float:
    """Estimate L = u_mean * T, T from an exp fit of the temporal autocorrelation.

    The autocorrelation of the point's signal is fit with exp(-lag / T) up to
    the first non-positive value (the classic truncated-autocorrelation method).
    """
    pt = inflow.data.loc[inflow.data["point_idx"] == point_idx].sort_values("time_step")
    signal = pt[component].to_numpy(dtype=float)
    times = pt["time_step"].to_numpy(dtype=float)
    signal = signal - signal.mean()
    if signal.std() == 0 or signal.size < 8:
        return float("nan")

    corr = np.correlate(signal, signal, mode="full")[signal.size - 1 :]
    corr = corr / corr[0]
    dt = float(np.mean(np.diff(times)))
    lags = np.arange(corr.size) * dt

    cut = np.argmax(corr <= 0.0)
    cut = corr.size if cut == 0 else cut
    if cut < 3:
        return float("nan")
    try:
        (tau,), _ = curve_fit(
            lambda x, t: np.exp(-x / t), lags[:cut], corr[:cut], p0=[lags[cut // 2] or dt]
        )
    except (RuntimeError, ValueError):
        return float("nan")
    return float(u_mean * tau)


# -- figures ---------------------------------------------------------------


def plot_mean_velocity(profile: ProfileLine, inflow: InflowData, *, component: str = "ux"):
    means = calculate_mean_velocity(inflow, for_components=[component])
    u = _profile_series(profile, means, f"{component}_mean")
    fig, ax = new_axes(
        xlabel=f"mean {component} [m/s]", ylabel="z [m]", title=f"Mean velocity -- {profile.name}"
    )
    ax.plot(u, profile.z, "-o", ms=3)
    return fig


def plot_turbulence_intensity(profile: ProfileLine, inflow: InflowData, *, component: str = "ux"):
    ti = calculate_turbulence_intensity(inflow, for_components=[component])
    iu = _profile_series(profile, ti, f"I_{component}")
    fig, ax = new_axes(
        xlabel=f"I_{component} [-]",
        ylabel="z [m]",
        title=f"Turbulence intensity -- {profile.name}",
    )
    ax.plot(iu, profile.z, "-o", ms=3)
    return fig


def plot_spectrum(
    profile: ProfileLine,
    inflow: InflowData,
    reference_height: float,
    norm: NormalizationParameters,
    *,
    component: str = "ux",
):
    target = profile.nearest_index(reference_height)
    spec = calculate_spectral_density(
        inflow, target_index=target, for_components=[component], normalization_params=norm
    )
    fig, ax = new_axes(
        xlabel="f L / U [-]",
        ylabel="f S(f) / sigma^2 [-]",
        title=f"Spectrum at z~{reference_height:g} m -- {profile.name}",
    )
    ax.loglog(spec[f"f ({component})"], spec[f"S ({component})"], lw=1.2)
    return fig
