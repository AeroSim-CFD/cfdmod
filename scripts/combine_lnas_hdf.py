import argparse
import pathlib
import sys
from dataclasses import dataclass

import pandas as pd
from lnas import LnasFormat

from cfdmod.api.vtk.write_vtk import create_polydata_for_cell_data, write_polydata
from cfdmod.use_cases.pressure.zoning.processing import calculate_statistics


@dataclass
class ArgsModel:
    """Command line arguments for client app"""

    lnas: str
    hdf: str
    output: str
    start_time: float
    end_time: float


def get_args_process(args: list[str]) -> ArgsModel:
    """Get arguments model from list of command line args

    Args:
        args (List[str]): List of command line arguments passed

    Returns:
        ArgsModel: Arguments model for client app
    """
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "-l",
        "--lnas",
        required=True,
        help="Path of lnas file",
        type=str,
    )
    ap.add_argument(
        "-hdf",
        "--hdf",
        required=True,
        help="Path of hdf time series data",
        type=str,
    )
    ap.add_argument(
        "-o",
        "--output",
        required=True,
        help="Output path for generated files",
        type=str,
    )
    ap.add_argument(
        "-st",
        "--start-time",
        help="Temporal range start time",
        default=0,
        type=float,
    )
    ap.add_argument(
        "-et",
        "--end-time",
        help="Temporal range end time",
        default=float("inf"),
        type=float,
    )

    parsed_args = ap.parse_args(args)
    args_model = ArgsModel(**vars(parsed_args))
    return args_model


def main(*args):
    args_use = get_args_process(*args)

    if args_use.end_time <= args_use.start_time:
        raise ValueError("The value for the end time must be bigger than the start time")

    lnas_path = pathlib.Path(args_use.lnas)
    hdf_path = pathlib.Path(args_use.hdf)
    output_path = pathlib.Path(args_use.output)

    mesh = LnasFormat.from_file(lnas_path)
    data = pd.read_hdf(hdf_path)

    if data.time_step.min() > args_use.end_time or data.time_step.max() < args_use.start_time:
        raise Exception("The temporal range does not include any time steps in the data")

    variables = [col for col in data.columns if col not in ["time_step", "point_idx"]]

    data = data.loc[
        (data["time_step"] >= args_use.start_time) & (data["time_step"] <= args_use.end_time)
    ]

    if data.point_idx.nunique() != len(mesh.geometry.triangles):
        raise ValueError("Number of points is different than number of triangles")

    stats = calculate_statistics(
        historical_data=data,
        statistics_to_apply=["max", "min", "std", "mean"],
        variables=variables,
        group_by_key="point_idx",
    )

    polydata = create_polydata_for_cell_data(data=stats, mesh=mesh.geometry)
    if not output_path.name.endswith(".vtp"):
        output_path = pathlib.Path(output_path.as_posix() + ".vtp")

    write_polydata(output_filename=output_path, poly_data=polydata)


if __name__ == "__main__":
    main(sys.argv[1:])
