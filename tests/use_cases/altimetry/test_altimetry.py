import pathlib

import numpy as np
import trimesh

from cfdmod.use_cases.altimetry import AltimetryProbe, AltimetrySection, Shed
from cfdmod.use_cases.altimetry.figure import savefig_to_file
from cfdmod.use_cases.altimetry.plots import plot_altimetry_profiles, plot_profiles, plot_surface


def test_image_generation():
    output_path = pathlib.Path("./output/altimetry")
    surface_mesh: trimesh.Trimesh = trimesh.load_mesh("./fixtures/tests/altimetry/terrain.stl")

    shed_start = np.array([-50, -50, 820], dtype=np.float32)
    shed_end = np.array([50, 50, 820], dtype=np.float32)

    shed = Shed(start_coordinate=shed_start, end_coordinate=shed_end, shed_label="default")

    altimetry_section = AltimetrySection.from_points("example", shed_start, shed_end)
    altimetry_section.slice_surface(surface_mesh)
    altimetry_section.include_shed(shed)

    fig, _ = plot_surface(surface_mesh, [altimetry_section])
    savefig_to_file(fig, output_path / "debug" / "surface.png")
    fig, _ = plot_profiles([altimetry_section])
    savefig_to_file(fig, output_path / "debug" / "profiles.png")
    fig, _ = plot_altimetry_profiles(altimetry_section)
    savefig_to_file(fig, output_path / "debug" / "altimetry.png")

    assert all(
        [
            (output_path / "debug" / f"{f}.png").exists()
            for f in ["altimetry", "profiles", "surface"]
        ]
    )


def test_probe_parsing():
    output_path = pathlib.Path("./output/altimetry")

    csv_path = pathlib.Path("./fixtures/tests/probes.csv")
    surface_mesh: trimesh.Trimesh = trimesh.load_mesh("./fixtures/tests/altimetry/terrain.stl")

    probes = AltimetryProbe.from_csv(csv_path)
    sections = np.unique([p.section_label for p in probes])
    altimetry_list: list[AltimetrySection] = []  # For debug plotting purposes

    assert len(sections) == 26

    for sec_label in sections[:5]:
        section_probes = [p for p in probes if p.section_label == sec_label]
        sheds_in_section = np.unique([p.building_label for p in section_probes])
        shed_list: list[Shed] = []

        for shed_label in sheds_in_section:
            building_probes = [p for p in section_probes if p.building_label == shed_label]
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
        [altimetry_section.include_shed(s) for s in shed_list]

        fig, _ = plot_altimetry_profiles(altimetry_section)
        savefig_to_file(fig, output_path / f"section-{altimetry_section.label}.png")
        altimetry_list.append(altimetry_section)

    assert all([(output_path / f"section-{s.label}.png").exists() for s in altimetry_list])


def test_slicing():
    surface_mesh: trimesh.Trimesh = trimesh.load_mesh("./fixtures/tests/altimetry/terrain.stl")

    plane_normal = np.array([1, 0, 0], dtype=np.float32)
    plane_origin = np.array([0, 0, 820], dtype=np.float32)

    altimetry_section = AltimetrySection("example", plane_origin, plane_normal)
    altimetry_section.slice_surface(surface_mesh)

    assert len(altimetry_section.section_vertices.pos) != 0
