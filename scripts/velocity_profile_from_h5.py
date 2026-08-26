"""Velocity-profile comparison, migrated from CSV to H5 input.

This is the H5-native version of the legacy "ExpAndSim_roof.py" post-processing
tool. It does the same thing the old CSV script did:

  1. read a line probe (point coordinates + a velocity component time series),
  2. average the velocity component over all time steps at each point,
  3. normalise the mean velocity by U_infinity and the height by B,
  4. optionally overlay a scaled experimental velocity profile,
  5. save the comparison figure (and the modified experimental CSV).

The only change from the original is the input: instead of reading
<line>.points.csv and <line>.ux.csv, it reads the coordinates from the
"Geometry" dataset and the velocity component from the matching field group
("ux") of a single line H5 file.

HOW TO USE: edit the CONFIG block right below, then run the file:

    python scripts/velocity_profile_from_h5.py
"""

from __future__ import annotations

import os

import h5py
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.ticker import FuncFormatter

# =============================================================================
# CONFIG -- edit these values, then just run the file.
# =============================================================================

# Input line-probe H5 file.
H5_PATH = r"/home/waine/Downloads/line.line_line_roof.h5"

# Velocity component to average ("ux", "uy" or "uz").
COMPONENT = "ux"

# Reference velocity used to normalise the mean profile.
U_INFINITY = 3.5

# Length B used to normalise the height z.
HEIGHT_SCALE = 40.0

# Where to write the outputs (figure + modified experimental CSV).
OUT_DIR = r"/home/waine/Downloads/h5_postpro_out"

# Optional experimental profile CSV to overlay. Set to None to skip the
# overlay and plot only the simulation curve.
EXPERIMENTAL_CSV = None
# EXPERIMENTAL_CSV = r"/home/waine/Downloads/velocity_profile_at_x_position_roof_AR1.csv"

# Experimental scaling (only used when EXPERIMENTAL_CSV is set), matching the
# original script: divide U by EXP_DIV_X, divide Y by EXP_DIV_Y, and keep only
# points with Y <= EXP_Y_MAX (after dropping the first point).
EXP_DIV_X = 1.3
EXP_DIV_Y = 1.2
EXP_Y_MAX = 6.0

# =============================================================================
# End of CONFIG. You normally do not need to edit below this line.
# =============================================================================


def mean_component_profile(h5_path: str, component: str) -> tuple[np.ndarray, np.ndarray]:
    """Return (z, time-mean component) along the line probe.

    The mean is accumulated one time step at a time so the full time series
    never has to sit in memory at once (a line H5 can hold thousands of
    steps). This reproduces pandas' ``ux.mean(axis=0)`` from the old script.
    """
    with h5py.File(h5_path, "r") as h5:
        if "Geometry" not in h5:
            raise KeyError(f"{h5_path}: no 'Geometry' dataset found")
        if component not in h5:
            raise KeyError(f"{h5_path}: no '{component}' field group found")

        z = h5["Geometry"][:, 2].astype(np.float64)

        group = h5[component]
        time_keys = list(group.keys())
        if not time_keys:
            raise ValueError(f"{h5_path}: field '{component}' has no time steps")

        accum = np.zeros(z.shape[0], dtype=np.float64)
        for key in time_keys:
            accum += group[key][:]
        mean_values = accum / len(time_keys)

    return z, mean_values


def load_experimental(path: str, div_x: float, div_y: float, y_max: float) -> pd.DataFrame:
    """Load and scale the experimental profile, matching the old script.

    Steps preserved from the original: drop the first point, keep only
    y <= ``y_max``, then divide U by ``div_x`` and Y by ``div_y``.
    """
    data = pd.read_csv(path)
    y = data["Y (normalized)"]
    u = data["U_normalized"]

    # Drop the first point (original behaviour).
    y = y[1:]
    u = u[1:]

    # Keep the near-ground portion of the profile.
    valid = y <= y_max
    y = y[valid]
    u = u[valid]

    # Apply the profile scaling factors.
    y = y / div_y
    u = u / div_x

    return pd.DataFrame({"Y (normalized)": y, "U_normalized": u})


def make_plot(
    z: np.ndarray,
    mean_u: np.ndarray,
    u_infinity: float,
    height_scale: float,
    experimental: pd.DataFrame | None,
    out_png: str,
) -> None:
    mean_u_norm = mean_u / u_infinity
    z_norm = z / height_scale

    # Sort by height so the simulated line plots cleanly.
    order = np.argsort(z_norm)
    z_norm = z_norm[order]
    mean_u_norm = mean_u_norm[order]

    fig, ax = plt.subplots(figsize=(5, 6))

    if experimental is not None:
        ax.plot(
            experimental["U_normalized"],
            experimental["Y (normalized)"],
            label="Experimental Data (U)",
            color="r",
            marker="o",
        )

    ax.plot(
        mean_u_norm,
        z_norm,
        label="Simulation Data (U)",
        color="b",
        linestyle="--",
    )

    ax.set_title("Comparison of Experimental and Simulation Velocity Profiles", fontsize=14)
    ax.set_xlabel(r"$\overline{u_x} / U_\infty$", fontsize=16)
    ax.set_ylabel(f"Normalized Y (z / {height_scale:g})", fontsize=16)
    ax.legend(loc="upper right", fontsize=12)
    ax.grid(True)

    ax.xaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:.1f}"))
    ax.set_xticks(np.linspace(mean_u_norm.min(), mean_u_norm.max(), 5))

    fig.tight_layout()
    fig.savefig(out_png, dpi=300)
    plt.close(fig)
    print(f"[INFO] plot saved: {out_png}")


def main() -> None:
    os.makedirs(OUT_DIR, exist_ok=True)

    base = os.path.basename(H5_PATH)
    if base.lower().endswith(".h5"):
        base = base[:-3]

    z, mean_u = mean_component_profile(H5_PATH, COMPONENT)
    print(
        f"[INFO] {COMPONENT}: {z.shape[0]} points, "
        f"mean range [{(mean_u / U_INFINITY).min():.3f}, "
        f"{(mean_u / U_INFINITY).max():.3f}] (normalised)"
    )

    experimental = None
    if EXPERIMENTAL_CSV:
        experimental = load_experimental(EXPERIMENTAL_CSV, EXP_DIV_X, EXP_DIV_Y, EXP_Y_MAX)
        exp_out = os.path.join(OUT_DIR, f"modified_experimental_{base}.csv")
        experimental.to_csv(exp_out, index=False)
        print(f"[INFO] modified experimental data saved: {exp_out}")

    out_png = os.path.join(OUT_DIR, f"velocity_profile_comparison_{base}.png")
    make_plot(z, mean_u, U_INFINITY, HEIGHT_SCALE, experimental, out_png)


if __name__ == "__main__":
    main()
