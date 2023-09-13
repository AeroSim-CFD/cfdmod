import argparse
import pathlib
from dataclasses import dataclass
from typing import List

import trimesh

from cfdmod.use_cases.altimetry import (
    AltimetryProbe,
    AltimetrySection,
    AltimetryShed,
    plot_altimetry_profiles,
)


@dataclass
class ArgsModel:
    """Command line arguments for client app"""

    csv: str
    surface: str
    output: str


def get_args_process(args: List[str]) -> ArgsModel:
    """Get arguments model from list of command line args

    Args:
        args (List[str]): List of command line arguments passed

    Returns:
        ArgsModel: Arguments model for client app
    """
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--csv",
        required=True,
        help="Path to probes csv table file",
        type=str,
    )
    ap.add_argument(
        "--surface",
        required=True,
        help="Path to stl surface file",
        type=str,
    )
    ap.add_argument(
        "--output",
        required=True,
        help="Output path",
        type=str,
    )
    parsed_args = ap.parse_args(args)
    args_model = ArgsModel(**vars(parsed_args))
    return args_model


def main(*args):
    args_use = get_args_process(*args)

    csv_path = pathlib.Path(args_use.csv)
    output_path = pathlib.Path(args_use.output)
    surface_mesh: trimesh.Trimesh = trimesh.load_mesh(args_use.surface)

    probes = AltimetryProbe.from_csv(csv_path)
    sections = set([p.section_label for p in probes])

    for sec_label in sections:
        section_probes = [p for p in probes if p.section_label == sec_label]
        sheds_in_section = set([p.building_label for p in section_probes])
        shed_list: list[AltimetryShed] = []

        for shed_label in sheds_in_section:
            building_probes = [p for p in section_probes if p.building_label == shed_label]
            shed = AltimetryShed(building_probes[0].coordinate, building_probes[1].coordinate)
            shed_list.append(shed)

        altimetry_section = AltimetrySection.from_points(
            sec_label, shed_list[0].start_coordinate, shed_list[0].end_coordinate
        )
        altimetry_section.slice_surface(surface_mesh)
        [altimetry_section.include_shed(s) for s in shed_list]

        plot_altimetry_profiles(altimetry_section, output_path)
