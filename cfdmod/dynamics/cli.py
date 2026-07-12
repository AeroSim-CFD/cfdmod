"""Typer CLI for the building-dynamics structural-import converters."""

from __future__ import annotations

import pathlib

import numpy as np
import typer

from cfdmod.dynamics.imports import read_eberick, read_tqs_portels
from cfdmod.dynamics.imports._csv_out import write_structural_csvs

app = typer.Typer(help="Building-dynamics utilities (structural-export conversion).")


@app.command("convert")
def convert(
    source: pathlib.Path = typer.Argument(
        ...,
        help="Export directory: TQS PORTELS(SE) .TXT files, or the Eberick "
        "DISTRIBUICAO + FORMAS_MODAIS .xlsx pair.",
    ),
    out: pathlib.Path = typer.Option(
        ..., "--out", "-o", help="Output directory for the internal modal CSVs."
    ),
    fmt: str = typer.Option("tqs", "--format", "-f", help="Source format: 'tqs' or 'eberick'."),
    active_modes: str | None = typer.Option(
        None,
        "--active-modes",
        help="Comma-separated 1-based mode numbers to keep (default: all).",
    ),
) -> None:
    """Convert a TQS/Eberick structural export to the internal modal CSVs.

    Writes ``modes.csv`` / ``floors.csv`` / ``phi{m}.csv`` under ``--out``,
    ready for ``BuildingStructuralData.from_csvs`` and the building dynamic
    recipe.
    """
    modes = [int(m) for m in active_modes.split(",")] if active_modes else None
    fmt = fmt.lower()
    try:
        if fmt == "tqs":
            sd = read_tqs_portels(source, active_modes=modes)
        elif fmt == "eberick":
            sd = read_eberick(source, active_modes=modes)
        else:
            raise ValueError(f"unknown --format {fmt!r}; expected 'tqs' or 'eberick'")
    except (FileNotFoundError, KeyError, ValueError) as exc:
        typer.echo(f"error: {exc}", err=True)
        raise typer.Exit(code=1) from exc

    paths = write_structural_csvs(sd, out)
    freqs = np.asarray(sd.natural_frequencies) / (2 * np.pi)
    typer.echo(
        f"converted {fmt} export '{source}': {sd.n_floors} floors, {sd.n_modes} modes "
        f"(f = {', '.join(f'{f:.3g}' for f in freqs)} Hz), "
        f"total mass {float(np.sum(sd.floors_mass)):.3g}."
    )
    typer.echo(f"wrote {len(paths)} files under {out}")


if __name__ == "__main__":
    app()
