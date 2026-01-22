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
