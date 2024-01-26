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


def get_args_process(args: list[str]) -> ArgsModel:
    """Get arguments model from list of command line args

    Args:
        args (List[str]): List of command line arguments passed

    Returns:
        ArgsModel: Arguments model for client app
    """
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--lnas",
        required=True,
        help="Path of lnas file",
        type=str,
    )
    ap.add_argument(
        "--hdf",
        required=True,
        help="Path of hdf time series data",
        type=str,
    )
    ap.add_argument(
        "--output",
        required=True,
        help="Output path for generated files",
        type=str,
    )

    parsed_args = ap.parse_args(args)
    args_model = ArgsModel(**vars(parsed_args))
    return args_model


def main(*args):
    args_use = get_args_process(*args)

    lnas_path = pathlib.Path(args_use.lnas)
    hdf_path = pathlib.Path(args_use.hdf)
    output_path = pathlib.Path(args_use.output)

    mesh = LnasFormat.from_file(lnas_path)
    data = pd.read_hdf(hdf_path)

    variables = [col for col in data.columns if col not in ["time_step", "point_idx"]]

    stats = calculate_statistics(
        historical_data=data,
        statistics_to_apply=["max", "min", "std", "mean"],
        variables=variables,
        group_by_key="point_idx",
    )
    polydata = create_polydata_for_cell_data(data=stats, mesh=mesh.geometry)

    write_polydata(output_filename=output_path, poly_data=polydata)


if __name__ == "__main__":
    main(sys.argv[1:])
