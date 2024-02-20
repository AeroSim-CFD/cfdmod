import argparse
from dataclasses import dataclass


@dataclass
class ArgsModel:
    """Command line arguments for client app"""

    vtp: str
    config: str
    output: str


def get_args_process(args: list[str]) -> ArgsModel:
    """Get arguments model from list of command line args

    Args:
        args (List[str]): List of command line arguments passed

    Returns:
        ArgsModel: Arguments model for client app
    """
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--vtp",
        required=True,
        help="Path for polydata file",
        type=str,
    )
    ap.add_argument(
        "--config",
        required=True,
        help="Path to config .yaml file",
        type=str,
    )
    ap.add_argument(
        "--output",
        required=True,
        help="Output path for generated images",
        type=str,
    )
    parsed_args = ap.parse_args(args)
    args_model = ArgsModel(**vars(parsed_args))
    return args_model


def main(*args):
    args_use = get_args_process(*args)
    projections = {
        "x_plus": [0, -90, 0],
        "x_minus": [0, 90, 0],
        "y_plus": [-90, 0, 0],
        "y_minus": [90, 0, 0],
    }

    COLORMAP_N_DIVS = 10
    OFFSET_VALUE = 15
