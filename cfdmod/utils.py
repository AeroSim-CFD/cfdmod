import pathlib


def create_folders_for_file(cls, filename: pathlib.Path):
    """Creates folders to save given file

    Args:
        filename (pathlib.Path): Filename to setup folder
    """

    filename.parent.mkdir(parents=True, exist_ok=True)


def create_folder_path(cls, path: pathlib.Path):
    """Creates folders path

    Args:
        path (pathlib.Path): Path to create
    """

    path.mkdir(parents=True, exist_ok=True)
