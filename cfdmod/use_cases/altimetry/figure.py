import pathlib

import matplotlib.pyplot as plt
from matplotlib.figure import Figure

from cfdmod.utils import create_folders_for_file


def savefig_to_file(fig: Figure, filename: pathlib.Path):
    """Creates folders to save given file
    
    Args:
        fig (Figure): Figure object to save
        filename (pathlib.Path): Filename to setup folder
    """
    create_folders_for_file(filename)
    fig.savefig(filename.as_posix())
    plt.close(fig)
