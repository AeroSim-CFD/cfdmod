"""Shared matplotlib helpers for the high-rise notebooks.

Kept deliberately small: a non-interactive backend so notebooks run headless
under nbconvert, one place to set a consistent figure style, and a helper to
close figures so long runs do not leak memory.
"""

from __future__ import annotations

import matplotlib

matplotlib.use("Agg")  # headless: notebooks write files, they do not display

import matplotlib.pyplot as plt  # noqa: E402


def apply_style() -> None:
    """Apply a consistent, readable style for all figures."""
    plt.rcParams.update(
        {
            "figure.figsize": (7.0, 5.0),
            "figure.dpi": 110,
            "axes.grid": True,
            "grid.alpha": 0.3,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "font.size": 11,
            "legend.frameon": False,
        }
    )


def new_axes(xlabel: str = "", ylabel: str = "", title: str = ""):
    """Return a fresh (fig, ax) with labels applied."""
    fig, ax = plt.subplots()
    if xlabel:
        ax.set_xlabel(xlabel)
    if ylabel:
        ax.set_ylabel(ylabel)
    if title:
        ax.set_title(title)
    return fig, ax


def close(fig) -> None:
    plt.close(fig)
