import argparse
import pathlib
from dataclasses import dataclass

from cfdmod.roughness.parameters import GenerationParams, RadialParams
from cfdmod.roughness.run import run_linear, run_radial


@dataclass
class ArgsModel:
    """Command line arguments for client app"""

    config: str
    output: str
    mode: str


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
        help="Path to config .yaml file",
        type=str,
    )
    ap.add_argument(
        "--output",
        required=True,
        help="Output path for stl file",
        type=str,
    )
    ap.add_argument(
        "--mode",
        default="linear",
        choices=["linear", "radial"],
        help="Generation mode: linear (default) or radial",
        type=str,
    )
    parsed_args = ap.parse_args(args)
    args_model = ArgsModel(**vars(parsed_args))
    return args_model


def main(*args):
    args_use = get_args_process(*args)
    output_path = pathlib.Path(args_use.output)

    if args_use.mode == "radial":
        cfg = RadialParams.from_file(pathlib.Path(args_use.config))
        run_radial(cfg, output_path)
        return

    cfg = GenerationParams.from_file(pathlib.Path(args_use.config))
    run_linear(cfg, output_path)
