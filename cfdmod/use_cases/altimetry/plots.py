import pathlib

import matplotlib.pyplot as plt
import numpy as np
import trimesh
from mpl_toolkits.mplot3d import Axes3D  # Needed for setting projection in figure

from cfdmod.use_cases.altimetry import AltimetrySection
from cfdmod.utils import create_folders_for_file


def plot_surface(
    surface: trimesh.Trimesh,
    altimetry_sections: list[AltimetrySection],
    output_path: pathlib.Path,
):
    """For debug: 3D plotting function that loads a mesh and receives the sections, plots and saves to file.

    Args:
        surface (trimesh.Trimesh): Trimesh object containing surface STL information.
        altimetry_sections (list[AltimetrySection]): List of altimetry sections.
        output_path (pathlib.Path): Path to the output directory for saving the images.
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

    for section in altimetry_sections:
        ax.plot(
            section.section_vertices.pos[:, 0],
            section.section_vertices.pos[:, 1],
            section.section_vertices.pos[:, 2],
            label=section.label,
            color=np.random.choice(range(256), size=3) / 255,
        )

    ax.set(xticklabels=[], yticklabels=[], zticklabels=[])
    ax.axis("off")
    fig.legend()

    # Save figure to output files
    filename = output_path / "debug" / "surface.png"
    create_folders_for_file(filename)
    fig.savefig(filename)
    plt.close()


def plot_profiles(altimetry_sections: list[AltimetrySection], output_path: pathlib.Path):
    """2D plotting function that receives section data from stl slicing, plots and saves to file.

    Args:
        altimetry_sections (list[AltimetrySection]): List of altimetry sections.
        output_path (pathlib.Path): Path to the output directory for saving the images.
    """
    for section in altimetry_sections:
        plt.plot(
            section.section_vertices.projected_position,
            section.section_vertices.pos[:, 2],
            color=np.random.choice(range(256), size=3) / 255,
            label=section.label,
        )

    # Save figure to output files
    plt.legend()
    filename = output_path / "debug" / "profiles.png"
    create_folders_for_file(filename)
    plt.savefig(filename)
    plt.close()


def plot_altimetry_profiles(altimetry_section: AltimetrySection, output_path: pathlib.Path):
    """2D plotting function to plot altimetry profiles and plot sheds from section data. Then it saves to a file

    Args:
        altimetry_section (AltimetrySection): Altimery section object containing section sheds (galpao) decomposed
        output_path (pathlib.Path): Path to the output directory for saving the images.
    """
    figure_heigth = (
        15
        * (
            (altimetry_section.section_vertices.maxz - altimetry_section.section_vertices.minz)
            / (
                max(altimetry_section.section_vertices.projected_position)
                - min(altimetry_section.section_vertices.projected_position)
            )
        )
        + 1.5
    )  # Figure height proportional to section profile aspect ratio, summed with an offset to account for axis labels
    fig = plt.figure(figsize=(15, figure_heigth))
    plt.subplots_adjust(bottom=0.35)

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
        ax.plot(shed.profile[0], shed.profile[1], color="r")

    filename = output_path / f"section-{altimetry_section.label}.png"
    create_folders_for_file(filename)

    ax.set_ylim([altimetry_section.section_vertices.minz, altimetry_section.section_vertices.maxz])
    ax.set_xlim([altimetry_section.section_vertices.minx, altimetry_section.section_vertices.maxx])

    ax.minorticks_on()
    ax.tick_params(axis="both", which="minor", labelsize=0)
    ax.set_xticks(
        np.arange(
            altimetry_section.section_vertices.minx,
            altimetry_section.section_vertices.maxx,
            step=20,
        ),
        minor=True,
    )
    ax.set_yticks(
        np.arange(
            altimetry_section.section_vertices.minz,
            altimetry_section.section_vertices.maxz + 1,
            step=50,
        )
    )
    ax.set_yticks(
        np.arange(
            altimetry_section.section_vertices.minz,
            altimetry_section.section_vertices.maxz + 1,
            step=10,
        ),
        minor=True,
    )

    ax.grid(which="minor", alpha=0.3, linestyle="dashed")
    ax.grid(which="major", alpha=1, linestyle="dashed")
    plt.xticks(
        np.arange(
            altimetry_section.section_vertices.minx,
            altimetry_section.section_vertices.maxx,
            step=100,
        ),
        rotation=90,
    )
    plt.title(f"Sec {altimetry_section.label}")
    plt.savefig(filename)
    plt.close()
