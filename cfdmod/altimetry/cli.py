"""Typer CLI for the altimetry module."""

from __future__ import annotations

import pathlib

import typer

from cfdmod.altimetry import AltimetryProbe, AltimetrySection, Shed
from cfdmod.altimetry.figure import savefig_to_file
from cfdmod.altimetry.plots import plot_altimetry_profiles

app = typer.Typer(name="altimetry", help="Altimetry section profile commands")


def _load_trimesh():
    """Import trimesh on demand, naming the extra when it is absent.

    Importing at module scope would make the whole `cfdmod` CLI -- including
    `cfdmod run`, which has nothing to do with altimetry -- unusable on an
    install that did not opt into the optional `geometry` extra.
    """
    try:
        import trimesh
    except ImportError as exc:
        raise ImportError(
            "cfdmod altimetry requires the optional 'geometry' extras. "
            "Install with: pip install aerosim-cfdmod[geometry]"
        ) from exc
    return trimesh


@app.command()
def main(
    csv: pathlib.Path = typer.Option(..., "--csv", help="Probe CSV table"),
    surface: pathlib.Path = typer.Option(..., "--surface", help="Terrain STL"),
    output: pathlib.Path = typer.Option(..., "--output", help="Output directory"),
) -> None:
    """Build altimetry section figures from probe + surface inputs."""
    trimesh = _load_trimesh()
    surface_mesh: trimesh.Trimesh = trimesh.load_mesh(surface.as_posix())

    probes = AltimetryProbe.from_csv(csv)
    sections = {p.section_label for p in probes}

    output.mkdir(parents=True, exist_ok=True)

    for sec_label in sections:
        section_probes = [p for p in probes if p.section_label == sec_label]
        sheds_in_section = {p.building_label for p in section_probes}
        shed_list: list[Shed] = []

        for shed_label in sheds_in_section:
            building_probes = sorted(
                [p for p in section_probes if p.building_label == shed_label],
                key=lambda x: (x.coordinate[0], x.coordinate[1]),
            )
            shed = Shed(
                start_coordinate=building_probes[0].coordinate,
                end_coordinate=building_probes[1].coordinate,
                shed_label=shed_label,
            )
            shed_list.append(shed)

        altimetry_section = AltimetrySection.from_points(
            sec_label, shed_list[0].start_coordinate, shed_list[0].end_coordinate
        )
        altimetry_section.slice_surface(surface_mesh)
        for s in shed_list:
            altimetry_section.include_shed(s)

        filename = output / f"section-{altimetry_section.label}.png"
        fig, _ = plot_altimetry_profiles(altimetry_section)
        savefig_to_file(fig, filename)
