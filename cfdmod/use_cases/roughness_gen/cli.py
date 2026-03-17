import pathlib
from typing import Annotated

import typer

from cfdmod.use_cases.roughness_gen.parameters import GenerationParams, RadialParams
from cfdmod.use_cases.roughness_gen.run import run_linear, run_radial

app = typer.Typer()


@app.command()
def main(
    config: Annotated[pathlib.Path, typer.Option(help="Path to config .yaml file")],
    output: Annotated[pathlib.Path, typer.Option(help="Output path for stl file")],
    mode: Annotated[str, typer.Option(help="Generation mode: linear or radial")] = "linear",
):
    if mode == "radial":
        cfg = RadialParams.from_file(config)
        run_radial(cfg, output)
    else:
        cfg = GenerationParams.from_file(config)
        run_linear(cfg, output)
