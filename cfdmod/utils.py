import pathlib

import matplotlib.pyplot as plt
from matplotlib.figure import Figure


def savefig_to_file(fig: Figure, filename: pathlib.Path):
    """Creates folders to save given file

    Args:
        fig (Figure): Figure object to save
        filename (pathlib.Path): Filename to setup folder
    """
    create_folders_for_file(filename)
    fig.savefig(filename.as_posix())
    plt.close(fig)


def create_folders_for_file(filename: pathlib.Path):
    """Creates folders to save given file

    Args:
        filename (pathlib.Path): Filename to setup folder
    """

    filename.parent.mkdir(parents=True, exist_ok=True)


def create_folder_path(path: pathlib.Path):
    """Creates folders path

    Args:
        path (pathlib.Path): Path to create
    """

    path.mkdir(parents=True, exist_ok=True)
