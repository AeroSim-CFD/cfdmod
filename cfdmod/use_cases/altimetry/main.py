import argparse
from dataclasses import dataclass
from typing import List


@dataclass
class ArgsModel:
    """Command line arguments for client app"""

    config: str


def get_args_process(args: List[str]) -> ArgsModel:
    """Get arguments model from list of command line args

    Args:
        args (List[str]): List of command line arguments passed

    Returns:
        ArgsModel: Arguments model for client app
    """
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--config",
        required=True,
        help="Path to config .yaml file",
        type=str,
    )
    parsed_args = ap.parse_args(args)
    args_model = ArgsModel(**vars(parsed_args))
    return args_model


def main(*args):
    args_use = get_args_process(*args)
    # Processing sequence:
    # Read surface file
    # Read probes list
    # Create surface object
    # Define sections
    # plane normal + plane origin
    # Create slice
    # Create shed points from probes
    # Plot slice and shed

    # Inputs:
    # - Probes list
    # - Surface file
    ...
