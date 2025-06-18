from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
import scipy
from scipy.ndimage import gaussian_filter

from cfdmod.use_cases.hfpi import solver


def plot_force_spectrum(
    forces_data: solver.HFPIForcesData,
    structure_data: solver.HFPIStructuralData,
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
        ax.loglog([df_mode["frequency"], df_mode["frequency"]], [1e-5, 1e1], color="black")
        ax.set_ylim(1e-4, 1e1)
        ax.set_xlim(1e-3, 3)

        ax.set_ylabel(r"$ S(F) f / \tilde{F}^2 $")
        ax.set_xlabel("f [Hz]")
        ax.legend(loc="lower left", frameon=False)

    return fig, ax


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

    return fig, ax


def set_plt_style():
    plt.style.use("seaborn-v0_8-whitegrid")
    plt.rcParams["font.family"] = "Ubuntu"
    plt.rcParams["font.size"] = 10
    plt.rcParams["mathtext.fontset"] = "custom"
    plt.rcParams["legend.facecolor"] = "white"
    plt.rcParams["legend.edgecolor"] = "none"
    plt.rcParams["xtick.direction"] = "in"
    plt.rcParams["ytick.direction"] = "in"
    plt.rcParams["lines.linestyle"] = "-"
    plt.rcParams["lines.linewidth"] = 2
    plt.rcParams["lines.markersize"] = 6
    plt.rcParams["lines.markeredgecolor"] = "none"
    plt.rcParams["axes.edgecolor"] = "black"
    plt.rcParams["figure.edgecolor"] = "red"
