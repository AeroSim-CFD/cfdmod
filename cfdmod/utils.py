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

import pathlib
from typing import Any

from ruamel.yaml import YAML


def read_yaml(filename: pathlib.Path) -> Any:
    """Read YAML from file

    Args:
        filename (str): File to read from

    Raises:
        Exception: Unable to read YAML from file

    Returns:
        Any: YAML content as python objects (dict, list, etc.)
    """
    if not filename.exists():
        raise Exception(f"Unable to read yaml. Filename {filename} does not exists")

    # Read YAML from file
    with open(filename, "r", encoding="utf-8") as f:
        try:
            yaml = YAML(typ="safe")
            return yaml.load(f)
        except Exception as e:
            raise Exception(
                f"Unable to load YAML from {filename}. Exception {e}"
            ) from e
