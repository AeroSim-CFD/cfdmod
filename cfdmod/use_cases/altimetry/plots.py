import matplotlib.pyplot as plt
import numpy as np
import trimesh
from matplotlib.figure import Figure

from cfdmod.use_cases.altimetry import AltimetrySection

__all__ = [
    "plot_surface",
    "plot_profiles",
    "plot_altimetry_profiles",
]


def plot_surface(
    surface: trimesh.Trimesh, altimetry_sections: list[AltimetrySection]
) -> tuple[Figure, plt.Axes]:
    """For debug: 3D plotting function that loads a mesh and receives the sections.

    Args:
        surface (trimesh.Trimesh): Trimesh object containing surface STL information.
        altimetry_sections (list[AltimetrySection]): List of altimetry sections.

    Returns:
        tuple[Figure, plt.Axes]: Tuple with figure and fig axis
    """
    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1, projection="3d")
    ax.plot_trisurf(
        surface.vertices[:, 0],
        surface.vertices[:, 1],
        surface.vertices[:, 2],
        triangles=surface.faces,
        color="#887321",
        linewidth=0.1,
    )
    ax.set_aspect("equal")

    color = np.random.choice(range(256), size=3).reshape(1, -1) / 255
    for section in altimetry_sections:
        ax.plot(
            section.section_vertices.pos[:, 0],
            section.section_vertices.pos[:, 1],
            section.section_vertices.pos[:, 2],
            label=section.label,
            color=color,
        )
        for sec_shed in section.section_sheds:
            ax.scatter(
                sec_shed.start_coordinate[0],
                sec_shed.start_coordinate[1],
                sec_shed.start_coordinate[2],
                c=color,
                s=2,
                marker="o",
                edgecolors="black",
                linewidths=1,
            )
            ax.scatter(
                sec_shed.end_coordinate[0],
                sec_shed.end_coordinate[1],
                sec_shed.end_coordinate[2],
                c=color,
                s=2,
                marker="o",
                edgecolors="black",
                linewidths=1,
            )

    ax.set(xticklabels=[], yticklabels=[], zticklabels=[])
    ax.axis("off")
    fig.legend()

    return (fig, ax)


def plot_profiles(altimetry_sections: list[AltimetrySection]) -> tuple[Figure, plt.Axes]:
    """2D plotting function that receives section data from stl slicing.

    Args:
        altimetry_sections (list[AltimetrySection]): List of altimetry sections.

    Returns:
        tuple[Figure, plt.Axes]: Tuple with figure and fig axis
    """
    fig = plt.figure()
    ax = fig.add_subplot()
    for section in altimetry_sections:
        ax.plot(
            section.section_vertices.projected_position,
            section.section_vertices.pos[:, 2],
            color=np.random.choice(range(256), size=3) / 255,
            label=section.label,
        )

    fig.legend()

    return (fig, ax)


def plot_altimetry_profiles(altimetry_section: AltimetrySection) -> tuple[Figure, plt.Axes]:
    """2D plotting function to plot altimetry profiles and plot sheds from section data.

    Args:
        altimetry_section (AltimetrySection): Altimery section object containing section sheds (galpao) decomposed

    Returns:
        tuple[Figure, plt.Axes]: Tuple with figure and fig axis
    """
    FIGURE_WIDTH = 15  # Constant found to generate figure as desired
    SCALING_CONSTANT = 1.5  # Constant found to generate figure as desired

    figure_height = (
        FIGURE_WIDTH
        * (
            (altimetry_section.section_vertices.maxz - altimetry_section.section_vertices.minz)
            / (
                max(altimetry_section.section_vertices.projected_position)
                - min(altimetry_section.section_vertices.projected_position)
            )
        )
        + SCALING_CONSTANT
    )  # Figure height proportional to section profile aspect ratio, summed with an offset to account for axis labels
    fig = plt.figure(figsize=(FIGURE_WIDTH, figure_height))
    fig.subplots_adjust(bottom=0.35)  # Padding to give figure some space

    ax = fig.add_subplot(111)
    ax.set_aspect("equal", "datalim")

    # Terrain profile plotting
    ax.plot(
        altimetry_section.section_vertices.projected_position,
        altimetry_section.section_vertices.pos[:, 2],
        color="b",
    )

    # Shed plotting
    for shed in altimetry_section.section_sheds:
        ax.plot(
            altimetry_section.section_shed_profiles[shed.shed_label][0],
            altimetry_section.section_shed_profiles[shed.shed_label][1],
            color="r",
            # color=(np.random.choice(range(156), size=3) + 100) / 255,
            # label=shed.shed_label, This logic can be later implemented
        )

    # If coloring logic for sheds is available, legend is necessary
    # fig.legend()

    ax.set_ylim(altimetry_section.section_vertices.minz, altimetry_section.section_vertices.maxz)
    ax.set_xlim(altimetry_section.section_vertices.minx, altimetry_section.section_vertices.maxx)

    ax.minorticks_on()
    ax.tick_params(axis="both", which="minor", labelsize=0)
    ax.set_xticks(
        np.arange(
            altimetry_section.section_vertices.minx,
            altimetry_section.section_vertices.maxx,
            step=20,  # Minor ticks should be 20 m apart in x axis
        ),
        minor=True,
    )
    ax.set_yticks(
        np.arange(
            altimetry_section.section_vertices.minz,
            altimetry_section.section_vertices.maxz + 1,
            step=50,  # Major ticks should be 50 m apart in y axis
        )
    )
    ax.set_yticks(
        np.arange(
            altimetry_section.section_vertices.minz,
            altimetry_section.section_vertices.maxz + 1,
            step=10,  # Minor ticks should be 10 m apart in y axis
        ),
        minor=True,
    )

    ax.grid(which="minor", alpha=0.3, linestyle="dashed")
    ax.grid(which="major", alpha=1, linestyle="dashed")
    plt.xticks(
        np.arange(
            altimetry_section.section_vertices.minx,
            altimetry_section.section_vertices.maxx,
            step=100,  # Major ticks should be 100 m apart in x axis
        ),
        rotation=90,
    )
    ax.set_title(f"Sec {altimetry_section.label}")

    return (fig, ax)
