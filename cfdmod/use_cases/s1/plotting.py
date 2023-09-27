import matplotlib.pyplot as plt


def set_style_fancy():
    plt.rcParams["font.family"] = "sans serif"  # "Calibri"
    plt.rcParams["font.size"] = 10
    plt.rcParams["mathtext.fontset"] = "custom"
    plt.rcParams["figure.facecolor"] = "none"
    plt.rcParams["axes.facecolor"] = "none"
    plt.rcParams["grid.color"] = "gainsboro"
    plt.rcParams["legend.facecolor"] = "white"
    plt.rcParams["legend.edgecolor"] = "none"
    plt.rcParams["lines.linestyle"] = ""
    plt.rcParams["lines.linewidth"] = 2
    plt.rcParams["lines.markersize"] = 6
    plt.rcParams["lines.markeredgecolor"] = "none"
    plt.rcParams["axes.grid.axis"] = "x"
    # plt.rcParams['axes.linewidth'] = 2.0
    plt.rcParams["grid.linewidth"] = 1.0


def set_style_tech():
    plt.rcParams["grid.color"] = "gray"
    plt.rcParams["font.family"] = "sans serif"  # "Calibri"
    plt.rcParams["font.size"] = 10
    plt.rcParams["mathtext.fontset"] = "custom"
    plt.rcParams["legend.facecolor"] = "white"
    plt.rcParams["legend.edgecolor"] = "none"
    plt.rcParams["lines.linestyle"] = ""
    plt.rcParams["lines.linewidth"] = 2
    plt.rcParams["lines.markersize"] = 6
    plt.rcParams["lines.markeredgecolor"] = "none"
    plt.rcParams["axes.edgecolor"] = "black"
    plt.rcParams["figure.edgecolor"] = "red"
