import pathlib
from typing import Dict

import matplotlib.pyplot as plt
import numpy as np
import trimesh
from mpl_toolkits.mplot3d import Axes3D

from cfdmod.use_cases.altimetry import SectionVertices
from cfdmod.utils import create_folders_for_file


def plot_surface(
    surface: trimesh.Trimesh,
    section_vertices: Dict[str, SectionVertices],
    output_path: pathlib.Path,
):
    """For debug: 3D plotting function that loads a mesh and receives the sections, plots and saves to file.

    Args:
        surface (trimesh.Trimesh): Trimesh object containing surface STL information.
        sections (Dict[str, SectionVertices]): Dictionary containing sections vertices.
        output_path (pathlib.Path): Path to the output directory for saving the images.
    """
    # Setup figure ref: https://stackoverflow.com/questions/59879666/plot-trimesh-object-like-with-axes3d-plot-trisurf
    figure = plt.figure()
    axes = figure.add_subplot(111, projection="3d")
    axes.plot_trisurf(
        surface.vertices[:, 0],
        surface.vertices[:, 1],
        surface.vertices[:, 2],
        triangles=surface.faces,
        color="#887321",
        shade=True,
    )
    axes.plot_trisurf()
    axes.auto_scale_xyz(1, 1, 1)  # Equal scaling

    for label, section in section_vertices.items():
        axes.plot(
            section.x,
            section.y,
            section.z,
            label=label,
            color=np.random.choice(range(256), size=3) / 255,
        )

    figure.add_axes(axes)
    plt.axis("off")
    plt.grid(b=None)

    # Save figure to output files
    filename = output_path / "debug" / "surface.png"
    create_folders_for_file(filename)
    plt.savefig(filename)
    plt.close()


def plot_profiles(sections: Dict[str, Section], output_path: str):
    """2D plotting function that receives section data from stl slicing, plots and saves to file.

    Args:
        sections (Dict[str, Section]): _description_
        output_path (str): Path to the output directory for saving the images.
    """
    for section in sections.values():
        plt.plot(
            section.vertices.projected_position,
            section.vertices.z,
            color=section.color,
            label=section.section,
        )
        plt.title(f"Seção {section.section}")

    # Save figure to output files
    plt.legend()
    filename = os.path.join(output_path, "debug", "profiles.png")
    create_folders_for_file(filename)
    plt.savefig(filename)
    plt.close()


def plot_altimetry_profiles(section: Section, output_path: str):
    """2D plotting function to plot altimetry profiles and plot sheds from section data, that contains probe objects. Then it saves to a file

    Args:
        section (Section): Section object containing section probes and sheds (galpao) decomposed
        output_path (str): Path to the output directory for saving the images.
    """
    figure_heigth = (
        15
        * (
            (section.vertices.maxz - section.vertices.minz)
            / (max(section.vertices.projected_position) - min(section.vertices.projected_position))
        )
        + 1.5
    )  # Figure height proportional to section profile aspect ratio, summed with an offset to account for axis labels
    fig = plt.figure(figsize=(15, figure_heigth))
    plt.subplots_adjust(bottom=0.35)

    ax = fig.add_subplot(111)
    ax.set_aspect("equal", "datalim")

    # Terrain profile plotting
    ax.plot(section.vertices.projected_position, section.vertices.z, color="b")

    # Shed plotting
    for shed in section.sheds:
        ax.plot(shed.profile[0], shed.profile[1], color="r")

    filename = os.path.join(output_path, f"section-{section.section}.png")
    create_folders_for_file(filename)

    ax.set_ylim([section.vertices.minz, section.vertices.maxz])
    ax.set_xlim([section.vertices.minx, section.vertices.maxx])

    ax.minorticks_on()
    ax.tick_params(axis="both", which="minor", labelsize=0)
    ax.set_xticks(np.arange(section.vertices.minx, section.vertices.maxx, step=20), minor=True)
    ax.set_yticks(np.arange(section.vertices.minz, section.vertices.maxz + 1, step=50))
    ax.set_yticks(np.arange(section.vertices.minz, section.vertices.maxz + 1, step=10), minor=True)

    ax.grid(which="minor", alpha=0.3, linestyle="dashed")
    ax.grid(which="major", alpha=1, linestyle="dashed")
    plt.xticks(np.arange(section.vertices.minx, section.vertices.maxx, step=100), rotation=90)
    plt.title(f"sec {section.section}")
    plt.savefig(filename)
    plt.close()
