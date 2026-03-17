import typer

from cfdmod.loft.cli import app as loft_app
from cfdmod.roughness.cli import app as roughness_app

app = typer.Typer()
app.add_typer(loft_app, name="loft")
app.add_typer(roughness_app, name="roughness")

if __name__ == "__main__":
    app()
