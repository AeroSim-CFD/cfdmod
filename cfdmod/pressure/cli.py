"""Typer CLI for the pressure module.

Each coefficient type is a subcommand: cp, cf, cm, ce. The ``--mesh`` option
is optional in every command -- when omitted, the geometry is read from the
input H5 (``--body`` for cp, ``--cp`` for cf/cm/ce). When provided, it accepts
``.lnas``, ``.stl``, ``.h5``, or ``.xdmf``; the loader at
:mod:`cfdmod.io.mesh` dispatches on the suffix.
"""

from __future__ import annotations

import pathlib
from typing import Optional

import typer

from cfdmod.pressure.run import run_ce, run_cf, run_cm, run_cp

app = typer.Typer(name="pressure", help="Pressure coefficient post-processing commands")

_MESH_HELP = (
    "Mesh file (.lnas/.stl/.h5/.xdmf). Optional; if omitted, the geometry is "
    "read from the input H5's embedded /Triangles + /Geometry."
)


@app.command("cp")
def cmd_cp(
    body: pathlib.Path = typer.Option(..., "--body", "-p", help="Body pressure XDMF+H5"),
    probe: Optional[pathlib.Path] = typer.Option(
        None, "--probe", "-s", help="Atmospheric probe XDMF+H5 (optional)"
    ),
    config: pathlib.Path = typer.Option(..., "--config", "-c", help="Cp YAML config"),
    output: pathlib.Path = typer.Option(..., "--output", "-o", help="Output directory"),
    mesh: Optional[pathlib.Path] = typer.Option(None, "--mesh", help=_MESH_HELP),
) -> None:
    """Compute pressure coefficient (Cp) timeseries and statistics."""
    run_cp(
        body_h5=body,
        probe_h5=probe,
        cfg_path=config,
        output=output,
        mesh_path=mesh,
    )


@app.command("cf")
def cmd_cf(
    cp: pathlib.Path = typer.Option(..., "--cp", help="Cp timeseries H5"),
    config: pathlib.Path = typer.Option(..., "--config", "-c", help="Cf YAML config"),
    output: pathlib.Path = typer.Option(..., "--output", "-o", help="Output directory"),
    mesh: Optional[pathlib.Path] = typer.Option(None, "--mesh", help=_MESH_HELP),
) -> None:
    """Compute force coefficient (Cf) and statistics."""
    run_cf(cp_h5=cp, cfg_path=config, output=output, mesh_path=mesh)


@app.command("cm")
def cmd_cm(
    cp: pathlib.Path = typer.Option(..., "--cp", help="Cp timeseries H5"),
    config: pathlib.Path = typer.Option(..., "--config", "-c", help="Cm YAML config"),
    output: pathlib.Path = typer.Option(..., "--output", "-o", help="Output directory"),
    mesh: Optional[pathlib.Path] = typer.Option(None, "--mesh", help=_MESH_HELP),
) -> None:
    """Compute moment coefficient (Cm) and statistics."""
    run_cm(cp_h5=cp, cfg_path=config, output=output, mesh_path=mesh)


@app.command("ce")
def cmd_ce(
    cp: pathlib.Path = typer.Option(..., "--cp", help="Cp timeseries H5"),
    config: pathlib.Path = typer.Option(..., "--config", "-c", help="Ce YAML config"),
    output: pathlib.Path = typer.Option(..., "--output", "-o", help="Output directory"),
    mesh: Optional[pathlib.Path] = typer.Option(None, "--mesh", help=_MESH_HELP),
) -> None:
    """Compute shape coefficient (Ce) and statistics."""
    run_ce(cp_h5=cp, cfg_path=config, output=output, mesh_path=mesh)
