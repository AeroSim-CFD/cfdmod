import matplotlib.pyplot as plt


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


plot_style = {
    "AeroSim": {
        "line": {"label": r"$\bf{AeroSim}$", "color": "#E69F00", "linestyle": "-"},
        "marker": {"color": "#E69F00", "markeredgewidth": 1.7, "linestyle": "none"},
    },
    "Exp": {
        "line": {"label": r"$\mathrm{EXP}$", "color": "#4d4d4d", "linestyle": "-"},
        "marker": {"color": "#4d4d4d", "markeredgewidth": 1.7, "linestyle": "none"},
    },
    "EU": {"line": {"label": "EN 1991-1-4", "color": "black", "linestyle": "--"}},
    "ABNT": {"line": {"label": "NBR 6123", "color": "black", "linestyle": ":"}},
}


def apply_style() -> None:
    """Apply a consistent, readable figure style for post-processing reports."""
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
    """Return a fresh (fig, ax) with the given labels applied."""
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


def set_style_tech() -> None:
    """Tech-report figure style: whitegrid, markers-only lines, thin spines.

    Meant for the code-comparison / deliverable figures (numerical points over
    analytical code curves), where a marker-only numerical series reads better
    than a connected line. Robust to matplotlib's seaborn-style rename.
    """
    for style in ("seaborn-v0_8-whitegrid", "seaborn-whitegrid"):
        if style in plt.style.available:
            plt.style.use(style)
            break
    plt.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.size": 10,
            "mathtext.fontset": "custom",
            "legend.facecolor": "white",
            "legend.edgecolor": "none",
            "lines.linestyle": "",
            "lines.linewidth": 2,
            "lines.markersize": 6,
            "lines.markeredgecolor": "none",
            "axes.edgecolor": "black",
        }
    )
