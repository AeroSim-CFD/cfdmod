__all__ = [
    "set_style_tech",
]


def set_style_tech(ax):
    # Set grid properties
    ax.grid(True, linestyle="-", linewidth=0.5, color="gray")

    # Set font properties
    for label in ax.get_xticklabels() + ax.get_yticklabels():
        label.set_fontsize(10)
        label.set_fontfamily("sans-serif")

    # Set math text fontset
    ax.xaxis.get_offset_text().set_fontsize(10)
    ax.yaxis.get_offset_text().set_fontsize(10)
    ax.title.set_fontsize(10)

    # Set legend properties
    legend = ax.get_legend()
    if legend:
        legend.get_frame().set_facecolor("white")
        legend.get_frame().set_edgecolor("none")

    # Set line properties
    for line in ax.lines:
        line.set_linestyle("solid")
        line.set_linewidth(2)
        line.set_markersize(6)
        line.set_markeredgecolor("none")

    # Set axes edge color
    ax.spines["bottom"].set_color("black")
    ax.spines["top"].set_color("black")
    ax.spines["right"].set_color("black")
    ax.spines["left"].set_color("black")
