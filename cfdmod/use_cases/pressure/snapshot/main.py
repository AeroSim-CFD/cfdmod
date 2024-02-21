import argparse
import pathlib
from dataclasses import dataclass

from cfdmod.use_cases.pressure.snapshot.camera import take_snapshot
from cfdmod.use_cases.pressure.snapshot.config import SnapshotConfig


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

    cfg_path = pathlib.Path(args_use.config)
    cfg = SnapshotConfig.from_file(cfg_path)

    output_path = pathlib.Path(args_use.output)
    vtp_path = pathlib.Path(args_use.vtp)

    for image_cfg in cfg.images:
        take_snapshot(
            scalar_name=image_cfg.scalar_label,
            file_path=vtp_path,
            output_path=output_path / f"{image_cfg.image_label}.png",
            colormap_params=cfg.colormap,
            project_params=cfg.projection,
            camera_params=cfg.camera,
        )
