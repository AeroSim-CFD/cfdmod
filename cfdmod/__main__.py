import typer

from cfdmod.altimetry.cli import app as altimetry_app
from cfdmod.loft.cli import app as loft_app
from cfdmod.pressure.cli import app as pressure_app
from cfdmod.regroup.cli import app as regroup_app
from cfdmod.roughness.cli import app as roughness_app

app = typer.Typer()
app.add_typer(altimetry_app, name="altimetry")
app.add_typer(loft_app, name="loft")
app.add_typer(pressure_app, name="pressure")
app.add_typer(regroup_app, name="regroup")
app.add_typer(roughness_app, name="roughness")

if __name__ == "__main__":
    app()
