import matplotlib.pyplot as plt
import numpy as np
from typing import Literal

from cfdmod.use_cases.s1 import profile
from cfdmod.use_cases.plot_config import plot_style


Languages = Literal["pt-br", "en"]


class SectionColors:
    colors: dict[str, str] = {
        "A": "#FF0000",
        "B": "#00FF00",
        "C": "#0000FF",
        "D": "#FF8000",
        "E": "#FF00FF",
        "F": "#00FFFF",
        "G": "#800080",
        "H": "#FF0080",
        "I": "#008080",
        "J": "#FF69B4",
        "K": "#800000",
        "L": "#000080",
        "M": "#FFA500",
        "N": "#008000",
        "O": "#800040",
        "P": "#9932CC",
        "Q": "#808040",
        "R": "#808080",
        "S": "#000000",
        "T": "#FF4500",
        "U": "#A52A2A",
        "V": "#4B0082",
        "W": "#C0C0C0",
        "X": "#8B008B",
        "Y": "#00008B",
        "Z": "#008040",
    }

    def get_section_color(self, section_label: str) -> str:
        if section_label in [k for k in self.colors.keys()]:
            return self.colors[section_label]
        else:
            return "#000000"

def set_style_tech():
    plt.style.use("seaborn-whitegrid")
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

def plot_s1(
    S1: dict[str, tuple[np.ndarray, np.ndarray]],
    direction_angle: float,
    shed_code: str,
    profile_height: dict[str, float],
):
    """Plots the processed S1 profiles per shed/block code

    Args:
        S1 (Dict[str, Tuple[np.ndarray, np.ndarray]]): S1 profiles (z,S1) keyed by probe label
        base_path (str): Path for saving the output image
        profile_height (Dict[str, float]): Dict with profile height keyed by probe label
        direction_angle (float): Direction angle of the case, defined by flow direction and -x axis
        shed_code (str): Shed/block code
    """

    direction_angle = round(direction_angle)

    plot_title = f"{shed_code}, {int(direction_angle)}°"
    set_style_tech()
    fig, ax = plt.subplots(figsize=(10 / 2.54, 10 / 2.54), constrained_layout=True, dpi=120)
    fig.suptitle(plot_title)

    section_colors: SectionColors = SectionColors()
    for probe in S1:
        x = S1[probe][0]
        y = S1[probe][1]
        secLabel = probe.split("-")[-1]
        color = section_colors.get_section_color(section_label=secLabel[-1])
        ax.plot(x, y, linestyle="-", label=f"Sec {secLabel}", color=color)

    # Limit offset
    maxS1 = max([max(S1[probe][0][3:]) for probe in S1])

    min_radius = 0.4
    max_radius = 0.6
    radius = abs(maxS1 - 1)
    if radius <= min_radius:
        radius = min_radius
    if radius > min_radius:
        radius = max_radius

    # sort both labels and handles by labels
    handles, labels = ax.get_legend_handles_labels()
    labels, handles = zip(*sorted(zip(labels, handles), key=lambda t: t[0]))

    ax.legend(handles, labels, loc="upper right", frameon=True)

    ax.set_xlabel(r"$\it{S}_1$")
    ax.set_ylabel(r"$\it{z} \; [m]$")

    ax.set_xticks(np.arange(1 - radius, 1 + radius + 0.0001, 0.2))
    ax.set_xlim(1 - radius, 1 + radius)
    ax.set_ylim(0, max([height for height in profile_height.values()]))
    return fig, ax
    

def plot_numerical_and_analytical_vel_profile(
    *,
    z: np.ndarray,
    H: float,
    u_num: np.ndarray,
    Iu_num: np.ndarray,
    u_num_ref: float,
    cat_eu: profile.EUCat | None,
    cat_nbr: profile.NBRCat | None,
    u_ref: float = 1,
    Fr: float = 0.65,
    language: Languages = "pt-br",
):
    arr_u_num = u_num / u_num_ref
    arr_Iu_num = Iu_num
    
    vel_title = {"pt-br": "Velocidade longitudinal média", "en": "Longitudinal average velocity"}
    turb_title = {"pt-br": "Intensidade turbulenta longitudinal", "en": "Longitudinal turbulent intensity"}

    fig, ax = plt.subplots(1, 2, figsize=(10, 5), layout="constrained", sharey=True)

    ax[0].plot(arr_u_num, z / H, **plot_style["AeroSim"]["line"])
    if(cat_eu is not None):
        arr_u_eu = profile.get_EU_cat_u_profile(z=z, H=H, cat=cat_eu, u_ref=u_ref, Fr=Fr)
        ax[0].plot(arr_u_eu, z / H, **plot_style["EU"]["line"])
    if(cat_nbr is not None):
        arr_u_nbr = profile.get_NBR_cat_u_profile(z=z, H=H, cat=cat_nbr, u_ref=u_ref, Fr=Fr)
        ax[0].plot(arr_u_nbr, z / H, **plot_style["ABNT"]["line"])
    ax[0].set_xlabel(r"$ \overline{u}_x / u_H$")
    ax[0].set_ylabel("$z/H$")
    # ax[0].legend(["AeroSim", "Experimental"], loc="lower right")
    ax[0].set_ylim(0, 1.5)
    ax[0].set_xlim(0, 1.4)
    ax[0].set_title(vel_title[language], fontsize=16)
    ax[0].legend(loc="best", frameon=False, fontsize=10)
    # ax[0].grid()

    ax[1].plot(arr_Iu_num, z / H, **plot_style["AeroSim"]["line"])
    if(cat_eu is not None):
        arr_Iu_eu = profile.get_EU_cat_Iu_profile(z=z, cat=cat_eu)
        ax[1].plot(arr_Iu_eu, z / H, **plot_style["EU"]["line"])
    ax[1].set_xlabel(r"$ ( \overline{u\prime u\prime} )^{1/2} / \overline{u}_x$")
    ax[1].set_ylim(0, 2)
    ax[1].set_xlim(0, 0.5)
    ax[1].set_title(turb_title[language], fontsize=16)
    # ax[1].grid()

    # fig.legend(loc="right")
    # plt.tight_layout(rect=[0, 0, 0.8, 1])
    # plt.show(fig)
    return fig, ax