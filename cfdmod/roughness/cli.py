import pathlib
from typing import Annotated

import typer

from cfdmod.roughness.parameters import GenerationParams, PositionParams, RadialParams
from cfdmod.roughness.run import run_linear, run_position, run_radial

app = typer.Typer()


@app.command()
def main(
    config: Annotated[pathlib.Path, typer.Option(help="Path to config .yaml file")],
    output: Annotated[pathlib.Path, typer.Option(help="Output path for stl file")],
    mode: Annotated[
        str, typer.Option(help="Generation mode: linear, radial or position")
    ] = "linear",
):
    if mode == "radial":
        cfg = RadialParams.from_file(config)
        run_radial(cfg, output)
    elif mode == "position":
        position_cfg = PositionParams.from_file(config)
        run_position(position_cfg, output)
    elif mode == "linear":
        generation_cfg = GenerationParams.from_file(config)
        run_linear(generation_cfg, output)
    else:
        raise typer.BadParameter(
            f"Unknown mode {mode!r}. Expected one of: linear, radial, position."
        )
