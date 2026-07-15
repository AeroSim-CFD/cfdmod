"""Standalone plotting / export for building dynamic-response results.

Presentation layer (outside the paradigm): these helpers render the v3
data sources produced by :mod:`cfdmod.core.recipes.dynamic` and the
:class:`~cfdmod.core.container.Container` of directional cases. They are
not pipeline ops. Ported from the former HFPI ``analysis`` module and
repointed at the v3 shapes -- floor fields are ``(n_floors, n_timesteps)``
(the legacy layout was ``(n_timesteps, n_floors)``), directional results
come from ``Container.join_by(lambda c: c.direction)``, and structural
metadata is passed as plain arrays (natural frequencies in Hz).

The module does not force a Matplotlib backend; callers / tests select one
(e.g. ``matplotlib.use("Agg")``).
"""

from __future__ import annotations

__all__ = [
    "plot_force_spectrum",
    "plot_displacement",
    "plot_global_stats_results",
    "plot_global_stats_per_direction",
    "get_xlims",
    "floor_load_xlims",
    "plot_floor_by_floor_mean_peaks",
    "export_global_stats_per_direction_csv",
    "plot_max_acceleration",
    "plot_acceleration_floor_by_floor",
    "effective_peak_loads_per_direction",
    "add_eberick_floor_height_and_pavements",
    "export_eberick_tables_to_xlsx",
]

import math
import pathlib
from typing import Literal

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy
from matplotlib.ticker import FuncFormatter
from scipy.ndimage import gaussian_filter

from cfdmod.core.container import Container
from cfdmod.core.data_source import PointsDataSource

plot_style = {
    "AeroSim": {
        "line": {"label": r"$\bf{AeroSim}$", "color": "#E69F00", "linestyle": "-"},
        "marker": {"color": "#E69F00", "markeredgewidth": 1.7, "linestyle": "none"},
    },
}

Languages = Literal["pt-br", "en"]

_COEF_STYLE = {"cf_x": ("FX", "blue"), "cf_y": ("FY", "red"), "cm_z": ("MZ", "grey")}

# N -> tonne-force divisor (1 tf = 1000 kgf * standard gravity 9.80665 m/s^2).
# Same exact value as cfdmod.building.loadcases._N_PER_TF, so the two modules'
# tonne-force deliverables agree to the last digit.
_N_PER_TF = 9806.65


def plot_force_spectrum(
    load_source: PointsDataSource,
    natural_frequencies_hz,
    *,
    fields: tuple[str, str, str] = ("cf_x", "cf_y", "cm_z"),
    plot_mz: bool = True,
    sigma: float = 2,
    ax=None,
):
    """Reduced power spectral density of the global (floor-summed) load.

    Args:
        load_source: Floor-load data source (fields ``cf_x`` / ``cf_y`` /
            ``cm_z``, each ``(n_floors, n_timesteps)``).
        natural_frequencies_hz: Modal natural frequencies (Hz) drawn as
            vertical markers.
        fields: The three load fields, in ``(FX, FY, MZ)`` order.
    """
    dt = float(load_source.time.timestep_size)
    if ax is None:
        fig, ax = plt.subplots(1, 1, figsize=(6, 4))
    else:
        fig = ax.figure

    for fld in fields:
        label, color = _COEF_STYLE.get(fld, (fld, "black"))
        if not plot_mz and fld == fields[2]:
            continue
        arr = np.asarray(load_source.fields.read(fld), dtype=np.float64)  # (n_floors, n_t)
        global_force = arr.sum(axis=0)
        freq, psd = scipy.signal.periodogram(global_force, 1 / dt, scaling="density")
        psd = psd * freq / (np.std(global_force) ** 2)
        ax.loglog(freq, gaussian_filter(psd, sigma=sigma), color=color, label=label, alpha=0.8)

    for f in np.atleast_1d(np.asarray(natural_frequencies_hz, dtype=np.float64)):
        ax.loglog([f, f], [1e-5, 1e1], color="orange", alpha=0.2)

    ax.set_ylim(1e-4, 1e1)
    ax.set_xlim(1e-3, 3)
    ax.set_ylabel(r"$ S(F) f / \tilde{F}^2 $")
    ax.set_xlabel("f [Hz]")
    ax.legend(loc="lower left", frameon=False)
    return fig, ax


def plot_displacement(
    response: PointsDataSource,
    *,
    floor: int,
    plot_limit: float,
    start_step_idx: int = 0,
    disp_x_field: str = "disp_x",
    disp_y_field: str = "disp_y",
):
    """Hodograph (x-y trace) of one floor's displacement, faded over time."""
    fig, ax = plt.subplots(1, 1, figsize=(6, 4))

    x_disp = np.asarray(response.fields.read(disp_x_field))[floor, start_step_idx:]
    y_disp = np.asarray(response.fields.read(disp_y_field))[floor, start_step_idx:]
    x_avg = x_disp.mean()
    y_avg = y_disp.mean()

    ax.set_xlim(-plot_limit + x_avg, plot_limit + x_avg)
    ax.set_ylim(-plot_limit + y_avg, plot_limit + y_avg)
    ax.set_ylabel("y [m]")
    ax.set_xlabel("x [m]")

    n_samples = len(x_disp)
    n_bins = 80
    max_alpha, min_alpha = 0.9, 0.1
    bin_size = max(n_samples // n_bins, 1)

    for n_bin in range(n_bins):
        alpha = min_alpha + n_bin / n_bins * (max_alpha - min_alpha)
        start = bin_size * n_bin
        end = bin_size * (n_bin + 1) if n_bin != n_bins - 1 else -1
        ax.plot(x_disp[start:end], y_disp[start:end], color="b", alpha=alpha)

    ax.plot([-10, 10], [0, 0], color="black", alpha=0.2)
    ax.plot([0, 0], [-10, 10], color="black", alpha=0.2)
    return fig, ax


def plot_global_stats_results(
    ax,
    df: pd.DataFrame,
    direction: Literal["x", "y", "z"],
    *,
    color: str,
    txt_lg: str,
    plot_mean: bool,
    plot_peaks: bool,
    unit_conversion: float = 1 / 1e6,
    language: Languages = "en",
    **plot_kwargs,
):
    """Plot min/max/mean of a global load component vs wind direction.

    ``df`` carries a ``direction`` column plus ``min_{d}`` / ``max_{d}`` /
    ``mean_{d}`` columns.
    """
    d = direction
    kwargs = plot_kwargs.copy()
    kwargs.update(
        fillstyle="none",
        color=color,
        markersize=6,
        markerfacecolor="none",
        markeredgecolor=color,
        markeredgewidth=1.5,
    )

    if plot_peaks:
        ax.plot(
            df["direction"],
            df[f"min_{d}"] * unit_conversion,
            "v:",
            label=f"Min ({txt_lg})",
            **kwargs,
        )
        ax.plot(
            df["direction"],
            df[f"max_{d}"] * unit_conversion,
            "^:",
            label=f"Max ({txt_lg})",
            **kwargs,
        )
    if plot_mean:
        mean_txt = "Mean" if language == "en" else "Media"
        ax.plot(df["direction"], df[f"mean_{d}"] * unit_conversion, "-", label=mean_txt, **kwargs)


def plot_global_stats_per_direction(
    stats_xis: dict[float, dict[str, pd.DataFrame]],
    unit_conversion: float = 1 / 1e6,
    unit_name: str = "MN",
    variable_types: list[Literal["static", "hfpi"]] = ["static", "hfpi"],
    xticks: float = 30,
    language: Languages = "pt-br",
    share_y: bool = True,
):
    """Grid of global force / moment stats vs wind direction, per damping ratio.

    With ``share_y`` (default) the two force panels (Fx, Fy) share one
    symmetric-about-zero y-axis and the two overturning-moment panels (Mx, My)
    share another, so magnitudes are directly comparable across components and
    the zero line sits at mid-height. The torsion panel (Mz) is scaled on its
    own symmetric axis. Uses the constrained layout engine so axis labels never
    overlap neighbouring panels.
    """
    fig, axs = plt.subplots(3, 2, figsize=(10, 12), layout="constrained")
    axs[2, 1].set_visible(False)

    stats_ex = next(iter(stats_xis.values()))
    directions = stats_ex["forces_static"]["direction"].to_numpy()
    max_dir = directions.max()

    color_static = "#333333"
    colors_eq = ["#E69F00", "#E66B00", "#BC00DD", "#0097DD"]
    uf = f"{unit_name}"
    um = f"{unit_name}.m"

    def style_axis(ax, *, scalar_name: str, unit: str, max_dir: float):
        ax.set_ylabel(f"{scalar_name} ({unit})", weight="bold")
        ax.set_xticks(np.arange(0, max_dir + 1, xticks))
        ax.set_xlim(-1, max_dir + 1)
        ax.xaxis.set_major_formatter(FuncFormatter(lambda val, _: rf"${val:g}^\circ$"))

    text_sta = "estatico" if language == "pt-br" else "static"
    text_dyn = "estatico equivalente" if language == "pt-br" else "static equivalent"

    kwargs = dict(plot_peaks=True, unit_conversion=unit_conversion)
    kwargs_dyn = [
        kwargs
        | dict(
            txt_lg=rf"{text_dyn}, $\xi={xi * 100:.2f}$%",
            plot_mean=False,
            color=colors_eq[i % len(colors_eq)],
            language=language,
            alpha=0.9,
        )
        for i, xi in enumerate(stats_xis.keys())
    ]
    kwargs_stat = kwargs | dict(
        txt_lg=f"{text_sta}", plot_mean=True, color=color_static, language=language
    )

    k = "forces_static"
    for d, ij in [("x", (0, 0)), ("y", (0, 1))]:
        style_axis(axs[ij], scalar_name=f"F{d}", unit=uf, max_dir=max_dir)
        axs[ij].axhline(y=0, color="gray", linewidth=1.5, alpha=0.7)
        if "static" in variable_types:
            plot_global_stats_results(axs[ij], stats_ex[k], d, **kwargs_stat)
        if "hfpi" in variable_types:
            for i, stats in enumerate(stats_xis.values()):
                plot_global_stats_results(axs[ij], stats[f"{k}_eq"], d, **kwargs_dyn[i])

    k = "moments_static"
    for d, ij in [("x", (1, 0)), ("y", (1, 1)), ("z", (2, 0))]:
        style_axis(axs[ij], scalar_name=f"M{d}", unit=um, max_dir=max_dir)
        axs[ij].axhline(y=0, color="gray", linewidth=1.5, alpha=0.7)
        if "static" in variable_types:
            plot_global_stats_results(axs[ij], stats_ex[k], d, **kwargs_stat)
        if "hfpi" in variable_types:
            for i, stats in enumerate(stats_xis.values()):
                plot_global_stats_results(axs[ij], stats[f"{k}_eq"], d, **kwargs_dyn[i])

    if share_y:
        # forces (Fx, Fy) share one symmetric axis; overturning moments (Mx, My)
        # share another; torsion (Mz) gets its own symmetric axis.
        _apply_symmetric_shared_ylim([axs[0, 0], axs[0, 1]])
        _apply_symmetric_shared_ylim([axs[1, 0], axs[1, 1]])
        _apply_symmetric_shared_ylim([axs[2, 0]])

    # legend in the empty bottom-right quadrant, so it never overlaps a panel
    handles, labels = axs[0, 0].get_legend_handles_labels()
    axs[2, 1].set_visible(True)
    axs[2, 1].axis("off")
    axs[2, 1].legend(handles, labels, loc="center", frameon=False)
    return fig, axs


def _sieve_round(extreme_val: float) -> float:
    """Round up to a clean tick multiple (1 / 5 / 10 / 50 by magnitude)."""
    if extreme_val < 1:
        sieve_size = 1
    elif extreme_val < 50:
        sieve_size = 5
    elif extreme_val < 100:
        sieve_size = 10
    else:
        sieve_size = 50
    return math.ceil(extreme_val / sieve_size) * sieve_size


def get_xlims(min_vals, max_vals) -> list[float]:
    """Symmetric, sieve-rounded x-limits enclosing the data (50% headroom)."""
    x_lim = _sieve_round(max(abs(min(min_vals)), max(max_vals)) * 1.5)
    return [-x_lim, x_lim]


def _apply_symmetric_shared_ylim(axes, headroom: float = 1.08) -> None:
    """Set one symmetric-about-zero y-limit, shared across ``axes``.

    The limit is the sieve-rounded largest absolute plotted value over all the
    axes, so paired panels (e.g. Fx / Fy) end up on the same scale with the
    zero line centred.
    """
    peak = 0.0
    for ax in axes:
        for line in ax.get_lines():
            ydata = np.asarray(line.get_ydata(), dtype=np.float64)
            if ydata.size:
                m = np.nanmax(np.abs(ydata))
                if np.isfinite(m):
                    peak = max(peak, float(m))
    if peak <= 0:
        return
    lim = _sieve_round(peak * headroom)
    for ax in axes:
        ax.set_ylim(-lim, lim)


def floor_load_xlims(vals_by_direction, *, unit_conversion: float = 1 / 1e6):
    """Shared symmetric x-limits across directions for the floor-by-floor plot.

    ``vals_by_direction`` is an iterable of ``vals_plot`` dicts (one per
    direction), each ``{"min"|"max"|"mean": {"x"|"y"|"z": array}}``. Returns
    ``[(lo, hi), (lo, hi), (lo, hi)]`` for the x/y/z panels, enclosing every
    direction, ready to hand to :func:`plot_floor_by_floor_mean_peaks` as
    ``x_lims`` so all figures share one comparable scale.
    """
    vals_by_direction = list(vals_by_direction)
    lims: list[tuple[float, float]] = []
    for comp in ("x", "y", "z"):
        mins = [float(np.min(vp["min"][comp])) * unit_conversion for vp in vals_by_direction]
        maxs = [float(np.max(vp["max"][comp])) * unit_conversion for vp in vals_by_direction]
        lo, hi = get_xlims(mins, maxs)
        lims.append((lo, hi))
    return lims


def plot_floor_by_floor_mean_peaks(
    *,
    vals_plot: dict[Literal["min", "max", "mean"], dict[Literal["x", "y", "z"], np.ndarray]],
    vals_labels: tuple[str, str, str],
    wind_dir: float,
    unit_conversion: float = 1 / 1e6,
    unit_name: str = "MN",
    y_abs: tuple[float, float] | None = None,
    x_lims: list[tuple[float, float]] | None = None,
    **plot_kwargs,
):
    """Per-floor profiles of mean + peak min/max for the three load components.

    ``x_lims`` fixes the x-axis limits for the three panels ``(x, y, z)`` as
    ``[(lo, hi), (lo, hi), (lo, hi)]``. Pass the same value to every direction
    (e.g. computed once across all directions) so the profiles are drawn on one
    shared, comparable scale. When ``None`` each panel is auto-scaled to its own
    data via :func:`get_xlims` (symmetric about zero).
    """
    min_vals, max_vals, mean_vals = vals_plot["min"], vals_plot["max"], vals_plot["mean"]
    color_use = "#DB9B10"

    fig, axs = plt.subplots(1, 3, figsize=(15, 5), layout="constrained", sharey="row")
    markers = ["o", "^", "v"]
    labels = [" (mean)", " (3s max)", " (3s min)"]

    for idx, (ax, component) in enumerate(zip(axs.flat, ("x", "y", "z"))):
        ax.axvline(x=0, color="gray", linewidth=1.5, alpha=0.7)
        if x_lims is not None:
            x_lim = x_lims[idx]
        else:
            x_lim = get_xlims(
                min_vals[component] * unit_conversion, max_vals[component] * unit_conversion
            )
        ax.set_xlim(x_lim[0], x_lim[1])
        for dct_data, mark, label_n in zip((mean_vals, max_vals, min_vals), markers, labels):
            ax.plot(
                dct_data[component] * unit_conversion,
                np.arange(0, len(dct_data[component])),
                marker=mark,
                label=r"$ \bf{AeroSim}$" + label_n,
                fillstyle="none",
                markeredgecolor=color_use,
                **plot_style["AeroSim"]["marker"],
            )

    if y_abs is not None:
        axs[0].set_ylim(y_abs[0], y_abs[1])

    axs[0].set_ylabel("Floor")
    axs[0].set_xlabel(f"{vals_labels[0]} ({unit_name})", weight="bold")
    axs[1].set_xlabel(f"{vals_labels[1]} ({unit_name})", weight="bold")
    axs[2].set_xlabel(f"{vals_labels[2]} ({unit_name}.m)", weight="bold")
    axs.flat[0].legend(loc="best", frameon=False, ncol=1, fontsize=10)
    fig.suptitle(rf"wind ${float(wind_dir):g}^\circ$", fontweight="bold")
    return fig, axs


def export_global_stats_per_direction_csv(csv_path: pathlib.Path, stats: dict[str, pd.DataFrame]):
    """Flatten per-direction global stats into a single CSV."""
    df = pd.DataFrame()
    df["direction"] = stats["forces_static"]["direction"].to_numpy()
    for k, df_use in stats.items():
        for col in df_use.columns:
            if col == "direction":
                continue
            df[f"{k}_{col}"] = df_use[col].to_numpy()
    df.to_csv(csv_path, index=None)


def plot_max_acceleration(
    max_ac: dict[float, float],
    natural_frequencies_hz,
    project_name: str = "AeroSim",
    unit_conversion: float = 1000 / 9.806,
    unit_name: str = "milli-g",
):
    """Peak acceleration per wind recurrence period vs comfort criteria.

    Comfort limits come from :mod:`cfdmod.building.comfort` (the single,
    source-verified home for NBR 6123 / NBCC / Melbourne acceleration limits);
    they return m/s^2 and are scaled here by ``unit_conversion`` to ``unit_name``.
    """
    from cfdmod.building.comfort import nbcc_acceleration_limit, nbr6123_acceleration_limit

    color_eq = "#E69F00"
    color_nbcc = "#2F993A"
    color_nbr_res = "#A82D2D"
    color_nbr_com = "#426AC2"

    freqs = np.atleast_1d(np.asarray(natural_frequencies_hz, dtype=np.float64))
    range_freq = [freqs.min(), min(freqs.max(), 1)]
    # limit decreases with frequency, so [hi f, lo f] -> [smaller, larger] bracket
    f_hi_lo = np.array([range_freq[1], range_freq[0]], dtype=np.float64)
    range_nbr_res = list(nbr6123_acceleration_limit(f_hi_lo, "residential") * unit_conversion)
    range_nbr_com = list(nbr6123_acceleration_limit(f_hi_lo, "commercial") * unit_conversion)
    range_nbcc = [
        nbcc_acceleration_limit("residential") * unit_conversion,
        nbcc_acceleration_limit("commercial") * unit_conversion,
    ]

    fig, ax = plt.subplots()

    def bracket(x_center, y_range, color, label):
        ax.plot(
            [x_center, x_center], y_range, "-", linewidth=4, label=label, color=color, alpha=0.8
        )
        for y in y_range:
            ax.plot(
                [x_center - 0.1, x_center + 0.1], [y, y], "-", linewidth=3, color=color, alpha=0.8
            )

    bracket(1.0, range_nbr_res, color_nbr_res, "NBR 6123 - residential")
    bracket(0.99, range_nbr_com, color_nbr_com, "NBR 6123 - comercial")
    bracket(10.0, range_nbcc, color_nbcc, "NBCC - residential and comercial")

    ax.plot(
        list(max_ac.keys()),
        np.array(list(max_ac.values())) * unit_conversion,
        "o",
        label=project_name,
        color=color_eq,
    )
    ax.set_ylabel(f"Acceleration [{unit_name}]")
    ax.set_xlabel("Wind recurrence period")
    ax.set_title("Maximum acceleration")
    ax.legend()
    return fig, ax


def plot_acceleration_floor_by_floor(
    acc: np.ndarray,
    natural_frequency_hz: float,
    rec_period: float,
    *,
    project_name: str = "AeroSim",
    unit_conversion: float = 1000 / 9.806,
    unit_name: str = "milli-g",
    last_floor: int | None = None,
    language: Languages = "en",
    standards_to_use: list[Literal["nbr_com", "nbr_res", "nbcc_res", "nbcc_com", "melbourne"]] = [
        "melbourne"
    ],
):
    """Per-floor peak acceleration profile vs the selected comfort standards."""
    color_eq = "#E69F00"
    colors = {
        "nbcc_res": "#2F993A",
        "nbcc_com": "#124711",
        "nbr_res": "#A82D2D",
        "nbr_com": "#426AC2",
        "melbourne": "#A20DDD",
    }
    texts = {
        "nbcc_res": {"en": "NBCC - residential", "pt-br": "NBCC - residencial"},
        "nbcc_com": {"en": "NBCC - comercial", "pt-br": "NBCC - comercial"},
        "nbr_res": {"en": "NBR 6123 - residential", "pt-br": "NBR 6123 - residencial"},
        "nbr_com": {"en": "NBR 6123 - comercial", "pt-br": "NBR 6123 - comercial"},
        "melbourne": {"en": "Melbourne (1992)", "pt-br": "Melbourne (1992)"},
        "lastfloor": {"en": "Last habitable floor", "pt-br": "Ultimo andar habitavel"},
    }

    compat = {"nbr_com": 1, "nbr_res": 1, "nbcc_res": 10, "nbcc_com": 10}
    valid = [s for s in standards_to_use if s == "melbourne" or compat.get(s) == rec_period]
    if not valid:
        valid = ["melbourne"]

    freq = float(natural_frequency_hz)
    fig, ax = plt.subplots()
    if last_floor is not None:
        ax.axhline(
            last_floor,
            color="grey",
            linestyle="--",
            label=texts["lastfloor"][language],
            linewidth=1,
        )

    # Comfort limits from cfdmod.building.comfort (source-verified home); they
    # return m/s^2 and are scaled here by unit_conversion to unit_name.
    from cfdmod.building.comfort import (
        melbourne1992_acceleration_limit,
        nbcc_acceleration_limit,
        nbr6123_acceleration_limit,
    )

    def criterion(std: str) -> float:
        if std == "nbr_com":
            return float(nbr6123_acceleration_limit(freq, "commercial")) * unit_conversion
        if std == "nbr_res":
            return float(nbr6123_acceleration_limit(freq, "residential")) * unit_conversion
        if std == "nbcc_res":
            return nbcc_acceleration_limit("residential") * unit_conversion
        if std == "nbcc_com":
            return nbcc_acceleration_limit("commercial") * unit_conversion
        # melbourne
        return float(melbourne1992_acceleration_limit(freq, rec_period)) * unit_conversion

    for std in valid:
        ax.axvline(criterion(std), label=texts[std][language], color=colors[std], linewidth=2)

    acc = np.asarray(acc, dtype=np.float64)
    ax.plot(acc * unit_conversion, np.arange(len(acc)), "o", label=project_name, color=color_eq)
    ax.legend()
    ax.set_xlim(0, ax.get_xlim()[1])
    ax.set_ylim(0, ax.get_ylim()[1])
    return fig, ax


def effective_peak_loads_per_direction(
    container: Container,
    *,
    feq_fields: tuple[str, str, str] = ("feq_x", "feq_y", "meq_z"),
    unit_conversion: float = 1 / _N_PER_TF,
    get_primary_load: tuple[bool, bool, bool] | bool = True,
) -> dict[str, pd.DataFrame]:
    """Per-direction, per-floor peak static-equivalent loads for export.

    ``container`` maps case parameters (with a ``direction`` attribute) to
    building-response data sources carrying the static-equivalent floor
    load fields. For each direction the dominant (larger-magnitude) peak is
    selected per axis; ``get_primary_load=False`` picks the opposite peak.

    Reads the ``feq_*`` fields directly (the dynamic static-equivalent floor
    loads), so this is the presentation-layer reducer the notebooks call for the
    Eberick / per-direction peak tables. For the *effective* envelope that also
    folds in the applied quasi-static loads (``fs_*`` / ``ms_z`` when attached by
    the fan-out driver), and for the full companion-load case generation, use
    :func:`cfdmod.building.loadcases.effective_load_stats` /
    :func:`cfdmod.building.loadcases.generate_load_cases`, which build on
    :func:`cfdmod.dynamics.cases.get_stats_forces_effective`.

    Exactly one case per direction is required: pre-filter the container
    (e.g. ``container.filter_by(lambda c: c.xi == xi)``) so each direction
    maps to a single case. A direction with more than one case raises
    ``ValueError`` rather than silently picking an arbitrary one.
    """
    if isinstance(get_primary_load, bool):
        get_primary_load = (get_primary_load, get_primary_load, get_primary_load)
    primary = dict(zip(("x", "y", "z"), get_primary_load))

    tables: dict[str, dict[str, np.ndarray]] = {"Fx": {}, "Fy": {}, "Mz": {}}
    for direction, sub in container.join_by(lambda c: c.direction).items():
        if len(sub) != 1:
            raise ValueError(
                f"direction {direction} maps to {len(sub)} cases; pre-filter the container "
                "to a single case per direction (e.g. one xi / recurrence period)"
            )
        resp = next(iter(sub.values()))
        for axis, field, load_name in zip(("x", "y", "z"), feq_fields, ("Fx", "Fy", "Mz")):
            arr = np.asarray(resp.fields.read(field), dtype=np.float64)  # (n_floors, n_t)
            fe_min = arr.min(axis=1)
            fe_max = arr.max(axis=1)
            if abs(fe_min.mean()) > abs(fe_max.mean()):
                selected = fe_min if primary[axis] else fe_max
            else:
                selected = fe_max if primary[axis] else fe_min
            tables[load_name][f"{float(direction):.1f}"] = selected * unit_conversion

    return {name: pd.DataFrame(cols) for name, cols in tables.items()}


def add_eberick_floor_height_and_pavements(
    df: pd.DataFrame, floor_height: np.ndarray, pavement_names: list[str]
):
    """Attach ``Cota`` (height) and ``Pavimento`` (floor name) columns."""
    if len(floor_height) != len(df) or len(floor_height) != len(pavement_names):
        raise ValueError("Incompatible array lengths to set floor height")
    df["Cota"] = floor_height
    df["Pavimento"] = pavement_names


def _is_float(s) -> bool:
    try:
        float(s)
        return True
    except ValueError:
        return False


def export_eberick_tables_to_xlsx(
    tables: dict[str, pd.DataFrame],
    filename: pathlib.Path,
    coordinate_system: Literal["global", "local"] = "global",
):
    """Export floor load tables to an Eberick-compatible xlsx workbook."""
    filename.parent.mkdir(parents=True, exist_ok=True)
    if coordinate_system == "global":
        tab_names = {"Fx": "Fx", "Fy": "Fy", "Mz": "Mz"}
    else:
        tab_names = {"Fx": "Forca vento", "Fy": "Forca transversal", "Mz": "Momento torsor"}

    with pd.ExcelWriter(filename, engine="openpyxl", mode="w") as writer:
        for key, df in tables.items():
            cols_order = ["Pavimento", "Cota"]
            cols_order += sorted([c for c in df.columns if _is_float(c)], key=lambda k: float(k))
            df.sort_values(by="Cota", ascending=False, inplace=True)
            df[cols_order].to_excel(writer, sheet_name=tab_names[key], index=False)
