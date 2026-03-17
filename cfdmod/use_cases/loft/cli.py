import pathlib

import lnas
import typer

from cfdmod.use_cases.loft.parameters import LoftCaseConfig
from cfdmod.use_cases.loft.run import run_loft

app = typer.Typer()


@app.command()
def main(
    config: pathlib.Path = typer.Option(..., help="Path to loft config file"),
    surface: pathlib.Path = typer.Option(..., help="Path to STL/LNAS surface file"),
    output: pathlib.Path = typer.Option(..., help="Output path"),
):
    cfg = LoftCaseConfig.from_file(config)
    geom = lnas.LnasFormat.from_file(surface).geometry
    run_loft(cfg, geom, output)
