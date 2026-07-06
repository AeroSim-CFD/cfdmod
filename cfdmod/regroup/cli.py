"""Typer CLI for the regroup pipeline."""

from __future__ import annotations

import pathlib

import typer

from cfdmod.regroup.parameters import RegroupConfig
from cfdmod.regroup.run import run_regroup

app = typer.Typer()


@app.command()
def main(
    config: pathlib.Path = typer.Option(..., help="Path to regroup config YAML."),
    geometry: pathlib.Path = typer.Option(
        ...,
        help="Input geometry (.lnas / .stl / .h5 / .xdmf).",
    ),
    timeseries: pathlib.Path = typer.Option(
        ...,
        help="Input HDF5 timeseries (rows = timesteps, columns = parent triangle ids).",
    ),
    output: pathlib.Path = typer.Option(
        ...,
        help="Output directory (created if missing).",
    ),
):
    """Split / rearrange triangles per the config and reorder the timeseries."""
    cfg = RegroupConfig.from_file(config)
    run_regroup(cfg, geometry, timeseries, output)
