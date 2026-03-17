"""Typer CLI for the pressure module.

Each coefficient type is a subcommand: cp, cf, cm, ce.
"""

from __future__ import annotations

import pathlib
from typing import Optional

import typer

from cfdmod.pressure.run import run_ce, run_cf, run_cm, run_cp

app = typer.Typer(name="pressure", help="Pressure coefficient post-processing commands")


@app.command("cp")
def cmd_cp(
    body: pathlib.Path = typer.Option(..., "--body", "-p", help="Body pressure H5"),
    probe: Optional[pathlib.Path] = typer.Option(
        None, "--probe", "-s", help="Atmospheric probe H5 (optional)"
    ),
    mesh: pathlib.Path = typer.Option(..., "--mesh", help="LNAS mesh file"),
    config: pathlib.Path = typer.Option(..., "--config", "-c", help="Cp YAML config"),
    output: pathlib.Path = typer.Option(..., "--output", "-o", help="Output directory"),
) -> None:
    """Compute pressure coefficient (Cp) timeseries and statistics."""
    run_cp(
        body_h5=body,
        probe_h5=probe,
        mesh_path=mesh,
        cfg_path=config,
        output=output,
    )


@app.command("cf")
def cmd_cf(
    cp: pathlib.Path = typer.Option(..., "--cp", help="Cp timeseries H5"),
    mesh: pathlib.Path = typer.Option(..., "--mesh", help="LNAS mesh file"),
    config: pathlib.Path = typer.Option(..., "--config", "-c", help="Cf YAML config"),
    output: pathlib.Path = typer.Option(..., "--output", "-o", help="Output directory"),
) -> None:
    """Compute force coefficient (Cf) and statistics."""
    run_cf(cp_h5=cp, mesh_path=mesh, cfg_path=config, output=output)


@app.command("cm")
def cmd_cm(
    cp: pathlib.Path = typer.Option(..., "--cp", help="Cp timeseries H5"),
    mesh: pathlib.Path = typer.Option(..., "--mesh", help="LNAS mesh file"),
    config: pathlib.Path = typer.Option(..., "--config", "-c", help="Cm YAML config"),
    output: pathlib.Path = typer.Option(..., "--output", "-o", help="Output directory"),
) -> None:
    """Compute moment coefficient (Cm) and statistics."""
    run_cm(cp_h5=cp, mesh_path=mesh, cfg_path=config, output=output)


@app.command("ce")
def cmd_ce(
    cp: pathlib.Path = typer.Option(..., "--cp", help="Cp timeseries H5"),
    mesh: pathlib.Path = typer.Option(..., "--mesh", help="LNAS mesh file"),
    config: pathlib.Path = typer.Option(..., "--config", "-c", help="Ce YAML config"),
    output: pathlib.Path = typer.Option(..., "--output", "-o", help="Output directory"),
) -> None:
    """Compute shape coefficient (Ce) and statistics."""
    run_ce(cp_h5=cp, mesh_path=mesh, cfg_path=config, output=output)
