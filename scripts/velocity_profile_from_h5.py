"""Velocity-profile comparison, migrated from CSV to H5 input.

This is the H5-native version of the legacy "ExpAndSim_roof.py" post-processing
tool. It does the same thing the old CSV script did:

  1. read a line probe (point coordinates + a velocity component time series),
  2. average the velocity component over all time steps at each point,
  3. normalise the mean velocity by U_infinity and the height by B,
  4. optionally overlay a scaled experimental velocity profile,
  5. save the comparison figure (and the modified experimental CSV).

The only change is the input: instead of reading

    <line>.points.csv   and   <line>.ux.csv

it reads the coordinates from the "Geometry" dataset and the velocity
component from the matching field group ("ux") of a single line H5 file.

Example (matches the original defaults: U_inf=3.5, B=40, ux)
------------------------------------------------------------
    python scripts/velocity_profile_from_h5.py \
        --h5 "line.line_line_roof.h5" \
        --u-infinity 3.5 --height-scale 40 \
        --experimental velocity_profile_at_x_position_roof_AR1.csv \
        --out-dir ./out
"""

from __future__ import annotations

import argparse
import os

import h5py
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.ticker import FuncFormatter


def _parse_time(key: str) -> float:
    return float(key.lstrip("t"))


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


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--h5", required=True, help="path to the line-probe H5 file")
    parser.add_argument(
        "--component",
        default="ux",
        help="velocity component field to average (default: ux)",
    )
    parser.add_argument(
        "--u-infinity",
        type=float,
        default=3.5,
        help="reference velocity used to normalise the mean profile (default: 3.5)",
    )
    parser.add_argument(
        "--height-scale",
        type=float,
        default=40.0,
        help="length B used to normalise the height z (default: 40)",
    )
    parser.add_argument(
        "--experimental",
        default=None,
        help="optional experimental profile CSV to overlay",
    )
    parser.add_argument("--exp-div-x", type=float, default=1.3, help="experimental U scale divisor")
    parser.add_argument("--exp-div-y", type=float, default=1.2, help="experimental Y scale divisor")
    parser.add_argument("--exp-y-max", type=float, default=6.0, help="max experimental Y kept")
    parser.add_argument(
        "--out-dir",
        default=".",
        help="directory to write outputs into (default: current dir)",
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    base = os.path.basename(args.h5)
    if base.lower().endswith(".h5"):
        base = base[:-3]

    z, mean_u = mean_component_profile(args.h5, args.component)
    print(
        f"[INFO] {args.component}: {z.shape[0]} points, "
        f"mean range [{(mean_u / args.u_infinity).min():.3f}, "
        f"{(mean_u / args.u_infinity).max():.3f}] (normalised)"
    )

    experimental = None
    if args.experimental:
        experimental = load_experimental(
            args.experimental, args.exp_div_x, args.exp_div_y, args.exp_y_max
        )
        exp_out = os.path.join(args.out_dir, f"modified_experimental_{base}.csv")
        experimental.to_csv(exp_out, index=False)
        print(f"[INFO] modified experimental data saved: {exp_out}")

    out_png = os.path.join(args.out_dir, f"velocity_profile_comparison_{base}.png")
    make_plot(z, mean_u, args.u_infinity, args.height_scale, experimental, out_png)


if __name__ == "__main__":
    main()
