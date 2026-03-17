import argparse
import pathlib
from dataclasses import dataclass

import lnas

from cfdmod.loft.parameters import LoftCaseConfig
from cfdmod.loft.run import run_loft


@dataclass
class ArgsModel:
    """Command line arguments for client app"""

    config: str
    surface: str
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
        "--config",
        required=True,
        help="Path to loft config file",
        type=str,
    )
    ap.add_argument(
        "--surface",
        required=True,
        help="Path to stl surface file",
        type=str,
    )
    ap.add_argument(
        "--output",
        required=True,
        help="Output path",
        type=str,
    )
    parsed_args = ap.parse_args(args)
    args_model = ArgsModel(**vars(parsed_args))
    return args_model


def main(*args):
    args_use = get_args_process(*args)
    cfg_file = pathlib.Path(args_use.config)
    mesh_path = pathlib.Path(args_use.surface)
    output_path = pathlib.Path(args_use.output)

    cfg = LoftCaseConfig.from_file(cfg_file)
    geom = lnas.LnasFormat.from_file(mesh_path).geometry
    run_loft(cfg, geom, output_path)
