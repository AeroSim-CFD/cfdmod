import pathlib

import typer
from pydantic import ValidationError

from cfdmod.altimetry.cli import app as altimetry_app
from cfdmod.dynamics.cli import app as dynamics_app
from cfdmod.loft.cli import app as loft_app
from cfdmod.recipes import run_yaml, status_yaml
from cfdmod.regroup.cli import app as regroup_app
from cfdmod.roughness.cli import app as roughness_app

app = typer.Typer()
app.add_typer(altimetry_app, name="altimetry")
app.add_typer(dynamics_app, name="dynamics", help="Convert TQS/Eberick structural exports.")
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
    skip_fresh: bool = typer.Option(
        False,
        "--skip-fresh",
        help="Skip recomputing outputs already up to date; run only what stale outputs need.",
    ),
    digest: str | None = typer.Option(
        None,
        "--digest",
        help="Override the input-digest strategy: size_mtime | content | backend.",
    ),
) -> None:
    """Execute a v3 pipeline template (cfdmod.core.pipeline_yaml).

    The template declares its own inputs, pipeline steps, and outputs;
    this command just loads the YAML and runs it via XdmfH5Storage.
    """
    try:
        bindings = run_yaml(
            template, output_root=output_root, skip_fresh=skip_fresh, digest=digest
        )
    except (KeyError, ValueError, FileNotFoundError, ValidationError) as exc:
        # These are user/config errors (bad template, missing file, typo'd
        # field). Show the message alone, not a multi-screen traceback.
        typer.echo(f"error: {exc}", err=True)
        raise typer.Exit(code=1) from exc
    if skip_fresh and not bindings:
        typer.echo(f"template '{template}': all outputs already fresh; nothing to do.")
    else:
        typer.echo(f"ran template '{template}': {len(bindings)} bindings produced.")


@app.command("status")
def status(
    template: pathlib.Path = typer.Argument(..., help="Path to a v3 pipeline YAML template."),
    output_root: pathlib.Path | None = typer.Option(
        None, "--output-root", help="Optional storage root override."
    ),
    digest: str | None = typer.Option(
        None, "--digest", help="Override the input-digest strategy."
    ),
) -> None:
    """Report per-output freshness (fresh | stale | missing) without running.

    Exits non-zero if any declared output is stale or missing, so the
    command is usable as a CI / Makefile freshness gate.
    """
    try:
        statuses = status_yaml(template, output_root=output_root, digest=digest)
    except (KeyError, ValueError, FileNotFoundError, ValidationError) as exc:
        typer.echo(f"error: {exc}", err=True)
        raise typer.Exit(code=1) from exc
    if not statuses:
        typer.echo(f"template '{template}': no declared outputs.")
        return
    any_dirty = False
    for name, st in statuses.items():
        if not st.is_fresh:
            any_dirty = True
        typer.echo(f"{st.status:>7}  {name}  ({st.reason})")
    raise typer.Exit(code=1 if any_dirty else 0)


if __name__ == "__main__":
    app()
