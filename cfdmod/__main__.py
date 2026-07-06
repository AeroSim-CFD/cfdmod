import pathlib

import typer
from pydantic import ValidationError

from cfdmod.altimetry.cli import app as altimetry_app
from cfdmod.loft.cli import app as loft_app
from cfdmod.recipes import run_yaml
from cfdmod.regroup.cli import app as regroup_app
from cfdmod.roughness.cli import app as roughness_app

app = typer.Typer()
app.add_typer(altimetry_app, name="altimetry")
app.add_typer(loft_app, name="loft", help="Generate terrain loft surfaces.")
app.add_typer(
    regroup_app, name="regroup", help="Split/reorder mesh triangles via a grouping chain."
)
app.add_typer(
    roughness_app, name="roughness", help="Generate roughness elements (linear / radial)."
)


@app.command("run")
def run(
    template: pathlib.Path = typer.Argument(..., help="Path to a v3 pipeline YAML template."),
    output_root: pathlib.Path | None = typer.Option(
        None,
        "--output-root",
        help="Optional storage root override. Defaults to filesystem root so the YAML paths resolve as-is.",
    ),
) -> None:
    """Execute a v3 pipeline template (cfdmod.core.pipeline_yaml).

    The template declares its own inputs, pipeline steps, and outputs;
    this command just loads the YAML and runs it via XdmfH5Storage.
    """
    try:
        bindings = run_yaml(template, output_root=output_root)
    except (KeyError, ValueError, FileNotFoundError, ValidationError) as exc:
        # These are user/config errors (bad template, missing file, typo'd
        # field). Show the message alone, not a multi-screen traceback.
        typer.echo(f"error: {exc}", err=True)
        raise typer.Exit(code=1) from exc
    typer.echo(f"ran template '{template}': {len(bindings)} bindings produced.")


if __name__ == "__main__":
    app()
