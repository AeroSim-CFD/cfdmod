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

For code validation, :func:`plot_profile_vs_code` overlays the simulated mean
velocity / turbulence intensity on the NBR 6123 and EN 1991-1-4 analytical
profiles, :func:`directional_reference_speed` returns the design U_H per wind
direction from an analytical wind profile, and :func:`eu_integral_length_scale`
gives the EN 1991-1-4 length-scale theory to compare against.
"""

from __future__ import annotations

import dataclasses
from typing import TYPE_CHECKING

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
from cfdmod.plot_config import new_axes

if TYPE_CHECKING:
    from cfdmod.analytical.wind_profile import WindProfile_EU, WindProfile_NBR
    from cfdmod.s1.plotting import Languages
    from cfdmod.s1.profile import EUCat, NBRCat


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


# -- code-standard comparison (NBR 6123 / EN 1991-1-4) ---------------------


def directional_reference_speed(
    wind_profile: "WindProfile_NBR | WindProfile_EU",
    *,
    height: float,
    recurrence_period: float = 50,
    directions: list[float] | None = None,
    use_kd: bool = False,
) -> pd.Series:
    """Design reference speed U_H per wind direction from an analytical profile.

    Thin loop over :meth:`WindProfile_NBR.get_U_H` / :meth:`WindProfile_EU.get_U_H`
    (built from a ``wind_analysis_{NBR,EU}.csv`` via their ``build`` classmethods).
    Returns a Series indexed by direction (degrees), sorted ascending; take
    ``.max()`` for the governing speed.
    """
    if directions is None:
        directions = wind_profile.directional_data["wind_direction"].tolist()
    speeds = {
        float(d): float(
            wind_profile.get_U_H(
                height=height,
                direction=float(d),
                recurrence_period=recurrence_period,
                use_kd=use_kd,
            )
        )
        for d in directions
    }
    return pd.Series(speeds, name="U_H").sort_index()


def plot_profile_vs_code(
    profile: ProfileLine,
    inflow: InflowData,
    reference_height: float,
    *,
    cat_eu: "EUCat | None" = None,
    cat_nbr: "NBRCat | None" = None,
    component: str = "ux",
    Fr: float = 0.65,
    language: "Languages" = "pt-br",
):
    """Mean velocity + turbulence intensity vs the NBR 6123 / EN 1991-1-4 curves.

    Delegates to :func:`cfdmod.s1.plotting.plot_numerical_and_analytical_vel_profile`,
    feeding it the simulated profile (mean ``u(z)``, ``Iu(z)`` and ``u_ref`` at
    ``reference_height``). Returns ``(fig, ax)`` where ``ax`` is the 2-panel array.
    """
    from cfdmod.s1.plotting import plot_numerical_and_analytical_vel_profile

    means = calculate_mean_velocity(inflow, for_components=[component])
    ti = calculate_turbulence_intensity(inflow, for_components=[component])
    u_num = _profile_series(profile, means, f"{component}_mean")
    iu_num = _profile_series(profile, ti, f"I_{component}")
    u_ref = float(np.interp(reference_height, profile.z, u_num))
    return plot_numerical_and_analytical_vel_profile(
        z=profile.z,
        H=reference_height,
        u_num=u_num,
        Iu_num=iu_num,
        u_num_ref=u_ref,
        cat_eu=cat_eu,
        cat_nbr=cat_nbr,
        Fr=Fr,
        language=language,
    )


def integral_length_scale_profile(
    inflow: InflowData,
    profile: ProfileLine,
    *,
    component: str = "ux",
) -> np.ndarray:
    """Integral length scale at each height of ``profile`` (NaN where unresolved)."""
    means = calculate_mean_velocity(inflow, for_components=[component])
    u = _profile_series(profile, means, f"{component}_mean")
    return np.array(
        [
            integral_length_scale(inflow, int(pi), float(um), component=component)
            for pi, um in zip(profile.point_idx, u)
        ]
    )


def eu_integral_length_scale(
    z: np.ndarray,
    z0: float,
    *,
    z_min: float = 1.0,
) -> np.ndarray:
    """EN 1991-1-4 (B.1) turbulence length scale L(z) = L_t (z/z_t)^alpha.

    L_t = 300 m, z_t = 200 m, alpha = 0.67 + 0.05 ln(z0); z is floored at
    ``z_min`` (below which L is held constant, per the code).
    """
    z_arr = np.asarray(z, dtype=float)
    alpha = 0.67 + 0.05 * np.log(z0)
    return 300.0 * (np.maximum(z_arr, z_min) / 200.0) ** alpha


def plot_integral_length_scale(
    z: np.ndarray,
    L_num: np.ndarray,
    H: float,
    *,
    L_theory: np.ndarray | None = None,
    language: "Languages" = "pt-br",
):
    """Numerical integral length scale (and optional theory) as L/H vs z/H."""
    title = {"pt-br": "Escala integral de comprimento", "en": "Integral length scale"}[language]
    fig, ax = new_axes(xlabel=r"$l_{int} / H$", ylabel="$z / H$", title=title)
    z_arr = np.asarray(z, dtype=float)
    ax.plot(np.asarray(L_num, dtype=float) / H, z_arr / H, "-o", ms=3, label="AeroSim")
    if L_theory is not None:
        ax.plot(np.asarray(L_theory, dtype=float) / H, z_arr / H, "--k", label="EN 1991-1-4")
    ax.legend(loc="best", frameon=False)
    return fig, ax
