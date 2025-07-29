from __future__ import annotations

import pathlib
from typing import Literal

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy
from matplotlib.ticker import FuncFormatter
from scipy.ndimage import gaussian_filter

from cfdmod.use_cases.hfpi import dynamic, handler, static

plot_style = {
    "AeroSim": {
        "line": {"label": r"$\bf{AeroSim}$", "color": "#E69F00", "linestyle": "-"},
        "marker": {"color": "#E69F00", "markeredgewidth": 1.7, "linestyle": "none"},
    },
}

Languages = Literal["pt-br", "en"]


def plot_force_spectrum(
    forces_data: dynamic.HFPIForcesData,
    structure_data: dynamic.HFPIStructuralData,
    *,
    plot_mz: bool = True,
    sigma: float = 2,
):
    fig, ax = plt.subplots(1, 1, figsize=(6, 4))
    dt = forces_data.delta_t
    df_mode = structure_data.df_modes
    coef_style = {"FX": "blue", "FY": "red", "MZ": "grey"}

    for f_name, df_force in [
        ("FX", forces_data.cf_x),
        ("FY", forces_data.cf_y),
        ("MZ", forces_data.cm_z),
    ]:
        if not plot_mz and f_name == "MZ":
            continue
        global_force = df_force.drop(columns=["time"]).sum(axis=1)

        (freq, PSD) = scipy.signal.periodogram(global_force, 1 / dt, scaling="density")
        PSD = PSD * freq / (np.std(global_force) ** 2)
        ax.loglog(
            freq,
            gaussian_filter(PSD, sigma=sigma),
            color=coef_style[f_name],
            label=f_name,
            alpha=0.8,
        )
        ax.loglog(
            [df_mode["frequency"], df_mode["frequency"]], [1e-5, 1e1], color="orange", alpha=0.2
        )
        ax.set_ylim(1e-4, 1e1)
        ax.set_xlim(1e-3, 3)

        ax.set_ylabel(r"$ S(F) f / \tilde{F}^2 $")
        ax.set_xlabel("f [Hz]")
        ax.legend(loc="lower left", frameon=False)

    return fig, ax


def plot_force_spectrum_np(
    ax,
    forces_dct: dict[str, np.ndarray],
    structure_data: dynamic.HFPIStructuralData,
    *,
    delta_t: float,
    plot_mz: bool = True,
    sigma: float = 2,
):
    dt = delta_t
    df_mode = structure_data.df_modes
    coef_style = {"FX": "blue", "FY": "red", "MZ": "grey"}

    for f_name, arr_force in [
        ("FX", forces_dct["x"]),
        ("FY", forces_dct["y"]),
        ("MZ", forces_dct["z"]),
    ]:
        if not plot_mz and f_name == "MZ":
            continue
        global_force = arr_force.sum(axis=1)

        (freq, PSD) = scipy.signal.periodogram(global_force, 1 / dt, scaling="density")
        PSD = PSD * freq / (np.std(global_force) ** 2)
        ax.loglog(
            freq,
            gaussian_filter(PSD, sigma=sigma),
            color=coef_style[f_name],
            label=f_name,
            alpha=0.8,
        )
        ax.loglog(
            [df_mode["frequency"], df_mode["frequency"]], [1e-5, 1e1], color="orange", alpha=0.2
        )
        ax.set_ylim(1e-4, 1e1)
        ax.set_xlim(1e-3, 3)

        ax.set_ylabel(r"$ S(F) f / \tilde{F}^2 $")
        ax.set_xlabel("f [Hz]")
        ax.legend(loc="lower left", frameon=False)

    return ax


def plot_displacement(
    displacement_dict: dict[str, np.ndarray],
    *,
    floor_plot: int,
    plot_limit: float,
    start_step_idx: int = 0,
):
    fig, ax = plt.subplots(1, 1, figsize=(6, 4))

    x_disp = displacement_dict["x"][start_step_idx:, floor_plot]
    y_disp = displacement_dict["y"][start_step_idx:, floor_plot]
    x_avg = x_disp.mean()
    y_avg = y_disp.mean()

    ax.set_xlim(-plot_limit + x_avg, plot_limit + x_avg)
    ax.set_ylim(-plot_limit + y_avg, plot_limit + y_avg)
    ax.set_ylabel("y [m]")
    ax.set_xlabel("x [m]")

    # Use differente alphas for plot
    n_samples = len(displacement_dict["x"])
    n_bins = 80
    max_alpha = 0.9
    min_alpha = 0.1
    bin_size = n_samples // 80

    for n_bin in range(n_bins):
        alpha = min_alpha + n_bin / n_bins * (max_alpha - min_alpha)
        start = bin_size * n_bin
        end = bin_size * (n_bin + 1) if n_bin != bin_size - 1 else -1

        ax.plot(
            x_disp[start:end],
            y_disp[start:end],
            color="b",
            alpha=alpha,
        )

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
    d = direction

    kwargs = plot_kwargs.copy()
    kwargs["fillstyle"] = "none"
    kwargs["color"] = color
    kwargs["markersize"] = 6
    kwargs["markerfacecolor"] = "none"
    kwargs["markeredgecolor"] = color
    kwargs["markeredgewidth"] = 1.5
    # txt_lg = "static equivalent" if is_dynamic else "static"

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
        mean_txt = "Mean" if language == "en" else "Média"
        ax.plot(df["direction"], df[f"mean_{d}"] * unit_conversion, "-", label=mean_txt, **kwargs)


def plot_global_stats_per_direction(
    stats_xis: dict[float, dict[str, pd.DataFrame]],
    unit_conversion: float = 1 / 1e6,
    unit_name: str = "MN",
    variable_types: list[Literal["static", "hfpi"]] = ["static", "hfpi"],
    xticks: float = 30,
    language: Languages = "pt-br",
):
    """Plot global values for statistics of results"""

    fig, axs = plt.subplots(3, 2, figsize=(10, 12), sharey="row")
    axs[2, 1].set_visible(False)

    stats_ex = next(iter(stats_xis.values()))
    directions = stats_ex["forces_static"]["direction"].to_numpy()
    max_dir = directions.max()

    color_static = "#333333"
    colors_eq = ["#E69F00", "#E66B00", "#BC00DD", "#0097DD"]

    k = "forces_static"
    uf = f"{unit_name}"
    um = f"{unit_name}.m"

    def style_global_stats_plot(
        fig,
        ax,
        *,
        scalar_name: str,
        unit: str,
        max_dir: float = 350,
    ):
        # ax.set_xlabel("Wind Direction (degrees)")
        ax.set_ylabel(f"{scalar_name} ({unit})", weight="bold")
        ax.set_xticks(np.arange(0, max_dir + 1, xticks))
        ax.set_xlim(0 - 1, max_dir + 1)
        ax.xaxis.set_major_formatter(FuncFormatter(lambda val, _: f"{val:.0f}°"))
        # ax.set_title(f"{scalar_name}", weight="bold")

    text_sta = "estático" if language == "pt-br" else "static"
    text_dyn = "estático equivalente" if language == "pt-br" else "static equivalent"

    kwargs = dict(plot_peaks=True, unit_conversion=unit_conversion)
    kwargs_dyn = [
        kwargs
        | dict(
            txt_lg=rf"{text_dyn}, $\xi={xi*100:.2f}$%",
            plot_mean=False,
            color=colors_eq[i],
            language=language,
            alpha=0.9,
        )
        for i, xi in enumerate(stats_xis.keys())
    ]
    kwargs_stat = kwargs | dict(
        txt_lg=f"{text_sta}", plot_mean=True, color=color_static, language=language
    )

    for d, ij in [("x", (0, 0)), ("y", (0, 1))]:
        style_global_stats_plot(fig, axs[ij], scalar_name=f"F{d}", unit=uf, max_dir=max_dir)
        axs[ij].axhline(y=0, color="gray", linewidth=1.5, alpha=0.7, linestyle="-")
        if "static" in variable_types:
            plot_global_stats_results(axs[ij], stats_ex[k], d, **kwargs_stat)
        if "hfpi" in variable_types:
            for i, (xi, stats) in enumerate(stats_xis.items()):
                plot_global_stats_results(axs[ij], stats[f"{k}_eq"], d, **kwargs_dyn[i])

    k = "moments_static"
    for d, ij in [("x", (1, 0)), ("y", (1, 1)), ("z", (2, 0))]:
        style_global_stats_plot(fig, axs[ij], scalar_name=f"M{d}", unit=um, max_dir=max_dir)
        axs[ij].axhline(y=0, color="gray", linewidth=1.5, alpha=0.7, linestyle="-")
        if "static" in variable_types:
            plot_global_stats_results(axs[ij], stats_ex[k], d, **kwargs_stat)
        if "hfpi" in variable_types:
            for i, (xi, stats) in enumerate(stats_xis.items()):
                plot_global_stats_results(axs[ij], stats[f"{k}_eq"], d, **kwargs_dyn[i])

    axs[2, 0].legend(loc="center left", bbox_to_anchor=(1.2, 0.5))
    axs[2, 0].plot([0, 360], [0, 0], color="black", alpha=0.2)

    return fig, axs


def plot_floor_by_floor_mean_peaks(
    *,
    # Tuple for 3 axis to plot, as ["min", "max", "mean"] = {"x": arr, "y": arr, "z": arr}
    vals_plot: dict[Literal["min", "max", "mean"], dict[Literal["x", "y", "z"], np.ndarray]],
    # Tuple as ("Cfx", "Cfy", "Cmz")
    vals_labels: tuple[str, str, str],
    wind_dir: float,
    x_lims: list[tuple[float, float]],
    unit_conversion: float = 1 / 1e6,
    unit_name: str = "MN",
    y_abs: tuple[float, float] | None,
    **plot_kwargs,
):
    min_vals = vals_plot["min"]
    max_vals = vals_plot["max"]
    mean_vals = vals_plot["mean"]

    color_use = "#DB9B10"

    fig, axs = plt.subplots(1, 3, figsize=(15, 5), layout="constrained", sharey="row")

    keys = ["mean", "max", "min"]
    markers = ["o", "^", "v"]
    labels = [" (media)", " (3s max)", " (3s min)"]

    for ax, component, x_lim in zip(axs.flat, ("x", "y", "z"), x_lims):
        ax.axvline(x=0, color="gray", linewidth=1.5, alpha=0.7, linestyle="-")
        ax.set_xlim(x_lim[0], x_lim[1])
        for dct_data, key, mark, label_n in zip(
            (mean_vals, max_vals, min_vals), keys, markers, labels
        ):
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

    axs[0].set_ylabel("Andar")
    axs[0].set_ylabel(f"{vals_labels[0]} ({unit_name})", weight="bold")
    axs[1].set_ylabel(f"{vals_labels[1]} ({unit_name})", weight="bold")
    axs[2].set_ylabel(f"{vals_labels[2]} ({unit_name}.m)", weight="bold")

    (axs.flat)[0].legend(loc="best", frameon=False, ncol=1, fontsize=10)

    fig.suptitle(f"vento {int(wind_dir)}°", fontweight="bold")
    plt.show()
    return fig, axs


def export_global_stats_per_direction_csv(csv_path: pathlib.Path, stats: dict[str, pd.DataFrame]):
    """Export stats generated by `HFPIFullResults.get_global_peaks_by_direction` to csv"""

    df = pd.DataFrame()
    directions = stats["forces_static"]["direction"].to_numpy()
    df["direction"] = directions

    for k in stats:
        df_use = stats[k]
        for col in df_use.columns:
            if col == "direction":
                continue
            k_col = f"{k}_{col}"
            df[k_col] = stats[k][col].to_numpy()
    df.to_csv(csv_path, index=None)


def plot_max_acceleration(
    max_ac: dict[float, float],
    structure_data: dynamic.HFPIStructuralData,
    project_name: str = "AeroSim",
    unit_conversion: float = 1000 / 9.806,
    unit_name: str = "milli-g",
):
    color_eq = "#E69F00"
    color_nbcc = "#2F993A"
    color_nbr_res = "#A82D2D"
    color_nbr_com = "#426AC2"

    range_freq = [
        structure_data.df_modes["frequency"].min(),
        min(structure_data.df_modes["frequency"].max(), 1),
    ]
    range_NBR_ac_residential = [
        0.01 * 4.08 * range_freq[1] ** -0.445 * unit_conversion,
        0.01 * 4.08 * range_freq[0] ** -0.445 * unit_conversion,
    ]
    range_NBR_ac_comertial = [
        0.01 * 6.12 * range_freq[1] ** -0.445 * unit_conversion,
        0.01 * 6.12 * range_freq[0] ** -0.445 * unit_conversion,
    ]
    range_NBCC = [15 * (9.806 / 1000) * unit_conversion, 25 * (9.806 / 1000) * unit_conversion]

    fig, ax = plt.subplots()

    ax.plot(
        [1, 1],
        range_NBR_ac_residential,
        "-",
        linewidth=4,
        label=f"NBR 6123 - residential",
        color=color_nbr_res,
        alpha=0.8,
    )
    ax.plot(
        [0.9, 1.1],
        [range_NBR_ac_residential[0], range_NBR_ac_residential[0]],
        "-",
        linewidth=3,
        color=color_nbr_res,
        alpha=0.8,
    )
    ax.plot(
        [0.9, 1.1],
        [range_NBR_ac_residential[1], range_NBR_ac_residential[1]],
        "-",
        linewidth=3,
        color=color_nbr_res,
        alpha=0.8,
    )

    ax.plot(
        [0.99, 0.99],
        range_NBR_ac_comertial,
        "-",
        linewidth=4,
        label=f"NBR 6123 - comercial",
        color=color_nbr_com,
        alpha=0.8,
    )
    ax.plot(
        [0.9, 1.1],
        [range_NBR_ac_comertial[0], range_NBR_ac_comertial[0]],
        "-",
        linewidth=3,
        color=color_nbr_com,
        alpha=0.8,
    )
    ax.plot(
        [0.9, 1.1],
        [range_NBR_ac_comertial[1], range_NBR_ac_comertial[1]],
        "-",
        linewidth=3,
        color=color_nbr_com,
        alpha=0.8,
    )

    ax.plot(
        [10, 10],
        range_NBCC,
        "-",
        linewidth=4,
        label=f"NBCC - residential and comercial",
        color=color_nbcc,
        alpha=0.8,
    )
    ax.plot(
        [9.9, 10.1], [range_NBCC[0], range_NBCC[0]], "-", linewidth=3, color=color_nbcc, alpha=0.8
    )
    ax.plot(
        [9.9, 10.1], [range_NBCC[1], range_NBCC[1]], "-", linewidth=3, color=color_nbcc, alpha=0.8
    )

    ax.plot(
        list(max_ac.keys()),
        np.array(list(max_ac.values())) * unit_conversion,
        "o",
        label=project_name,
        color=color_eq,
    )

    ax.set_ylabel(f"Acceleration [{unit_name}]")
    ax.set_xlabel("Wind recurrence period")
    ax.set_title(f"Maximum acceleration")
    ax.legend()

    return fig, ax


def plot_acceleration_floor_by_floor(
    acc: np.ndarray,
    structure_data: dynamic.HFPIStructuralData,
    *,
    project_name: str = "AeroSim",
    unit_conversion: float = 1000 / 9.806,
    unit_name: str = "milli-g",
    language="en",
    plot_nbcc: bool = False,
    plot_nbr: bool = False,
    plot_melbourne: bool = False,
    melbourne_years: int = 10,
):
    color_eq = "#E69F00"
    color_nbcc = "#2F993A"
    color_nbr_res = "#A82D2D"
    color_nbr_com = "#426AC2"
    color_melbourne = "#A20DDD"

    fig, ax = plt.subplots()

    range_freq = [
        structure_data.df_modes["frequency"].min(),
        min(structure_data.df_modes["frequency"].max(), 1),
    ]

    texts = {
        "nbcc": {
            "en": "NBCC - residential and comercial",
            "pt-br": "NBCC - residencial e comercial",
        },
        "nbr_res": {"en": "NBR 6123 - residential", "pt-br": "NBR 6123 - residencial"},
        "nbr_com": {"en": "NBR 6123 - comercial", "pt-br": "NBR 6123 - comercial"},
        "melbourne": {"en": "Melbourne (1992)", "pt-br": "Melbourne (1992)"},
    }

    kwargs_codes = dict(alpha=0.15, linewidth=2)
    if plot_nbr:
        range_NBR_ac_residential = [
            0.01 * 4.08 * range_freq[1] ** -0.445 * unit_conversion,
            0.01 * 4.08 * range_freq[0] ** -0.445 * unit_conversion,
        ]
        range_NBR_ac_comertial = [
            0.01 * 6.12 * range_freq[1] ** -0.445 * unit_conversion,
            0.01 * 6.12 * range_freq[0] ** -0.445 * unit_conversion,
        ]
        ax.axvspan(
            range_NBR_ac_comertial[0],
            range_NBR_ac_comertial[1],
            label=texts["nbr_com"][language],
            color=color_nbr_com,
            **kwargs_codes,
        )
        ax.axvspan(
            range_NBR_ac_residential[0],
            range_NBR_ac_residential[1],
            label=texts["nbr_res"][language],
            color=color_nbr_res,
            **kwargs_codes,
        )
    if plot_nbcc:
        range_NBCC = [
            15 * (9.806 / 1000) * unit_conversion,
            25 * (9.806 / 1000) * unit_conversion,
        ]
        ax.axvspan(
            range_NBCC[0],
            range_NBCC[1],
            label=texts["nbcc"][language],
            color=color_nbcc,
            **kwargs_codes,
        )
    if plot_melbourne:
        range_melbourne = [
            unit_conversion
            * np.sqrt(2 * np.log(600 * range_freq[1]))
            * (0.68 + np.log(melbourne_years) / 5)
            * np.exp(-3.65 - 0.41 * np.log(range_freq[1])),
            unit_conversion
            * np.sqrt(2 * np.log(600 * range_freq[0]))
            * (0.68 + np.log(melbourne_years) / 5)
            * np.exp(-3.65 - 0.41 * np.log(range_freq[0])),
        ]
        ax.axvspan(
            range_melbourne[0],
            range_melbourne[1],
            label=texts["melbourne"][language],
            color=color_melbourne,
            **kwargs_codes,
        )
    pavements = np.arange(len(acc))
    ax.plot(acc * unit_conversion, pavements, "o", label=project_name, color=color_eq)
    ax.legend()
    ax.set_xlim(0, ax.get_xlim()[1])
    ax.set_ylim(0, ax.get_ylim()[1])

    return fig, ax


def get_effective_forces_peak_loads_per_direction(
    results: handler.HFPIAnalysisResults, unit_conversion: float = 1 / 9806
) -> dict[str, pd.DataFrame]:
    tables = {"Fx": {}, "Fy": {}, "Mz": {}}
    for wd, res in results.join_by_direction().items():
        r = next(iter(res.results.values()))
        fe_min = r.get_stats_forces_effective("min")
        fe_max = r.get_stats_forces_effective("max")
        for axis, load_axis in zip(["x", "y", "z"], ["Fx", "Fy", "Mz"]):
            if abs(fe_min[axis].mean()) > abs(fe_max[axis].mean()):
                stat_selection = fe_min[axis]
            else:
                stat_selection = fe_max[axis]
            tables[load_axis][f"{int(wd)}"] = stat_selection * unit_conversion
    tables_df: dict[str, pd.DataFrame] = {}
    for t in tables:
        tables_df[t] = pd.DataFrame(tables[t])
    return tables_df


def add_eberick_floor_height_and_pavements(
    df: pd.DataFrame, floor_height: np.ndarray, pavement_names: list[str]
):
    if len(floor_height) != len(df) or len(floor_height) != len(pavement_names):
        raise ValueError("Incompatible array lengths to set floor height")
    df["Cota"] = floor_height
    df["Pavimento"] = pavement_names


def export_eberick_tables_to_xlsx(tables: dict[str, pd.DataFrame], filename: pathlib.Path):
    """Export floor values to Eberick compatible xlsx

    https://suporte.altoqi.com.br/hc/pt-br/articles/360050991093"""

    filename.parent.mkdir(parents=True, exist_ok=True)
    tab_names = {"Fx": "Força vento", "Fy": "Força transversal", "Mz": "Momento torsor"}

    with pd.ExcelWriter(filename, engine="openpyxl") as writer:
        for key, df in tables.items():
            cols_order = ["Pavimento", "Cota"]
            cols_order += sorted([c for c in df.columns if c.isnumeric()], key=lambda k: int(k))
            df[cols_order].to_excel(writer, sheet_name=tab_names[key], index=False)
