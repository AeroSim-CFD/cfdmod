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
