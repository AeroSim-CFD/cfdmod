import argparse
import pathlib
from dataclasses import dataclass

from nassu.lnas import LagrangianFormat

from cfdmod.api.vtk.write_vtk import create_polydata_for_cell_data, write_polydata
from cfdmod.use_cases.pressure.cp_config import CpConfig
from cfdmod.use_cases.pressure.cp_data import (
    calculate_statistics,
    read_pressure_data,
    transform_to_cp,
)


@dataclass
class ArgsModel:
    """Command line arguments for client app"""

    output: str
    p: str
    s: str
    mesh: str
    config: str


def get_args_process(args: list[str]) -> ArgsModel:
    """Get arguments model from list of command line args

    Args:
        args (List[str]): List of command line arguments passed

    Returns:
        ArgsModel: Arguments model for client app
    """
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--output",
        required=True,
        help="Output path for generated files",
        type=str,
    )
    ap.add_argument(
        "--p",
        required=True,
        help="Path to body pressure series .hdf",
        type=str,
    )
    ap.add_argument(
        "--s",
        required=True,
        help="Path to static pressure series .hdf",
        type=str,
    )
    ap.add_argument(
        "--mesh",
        required=True,
        help="Path to LNAS mesh file",
        type=str,
    )
    ap.add_argument(
        "--config",
        required=True,
        help="Path to config .yaml file",
        type=str,
    )
    parsed_args = ap.parse_args(args)
    args_model = ArgsModel(**vars(parsed_args))
    return args_model


def main(*args):
    args_use = get_args_process(*args)
    post_proc_cfg = CpConfig.from_file(pathlib.Path(args_use.config))
    mesh = LagrangianFormat.from_folder(pathlib.Path(args_use.mesh))
    output_path = pathlib.Path(args_use.output)

    static_data_path = pathlib.Path(args_use.s)
    body_data_path = pathlib.Path(args_use.p)

    press_data, body_data = read_pressure_data(
        static_data_path, body_data_path, post_proc_cfg.timestep_range
    )

    # OUTPUT 1: cp(t)
    cp_data = transform_to_cp(
        press_data,
        body_data,
        reference_vel=post_proc_cfg.U_H,
        ref_press_mode=post_proc_cfg.reference_pressure,
    )
    cp_data.to_hdf(output_path / "cp_t.hdf", key="cp_t", mode="w")

    # OUTPUT 2: cp_stats
    cp_stats = calculate_statistics(cp_data, statistics_to_apply=post_proc_cfg.statistics)
    cp_stats.to_hdf(output_path / "cp_stats.hdf", key="cp_t", mode="w")

    polydata = create_polydata_for_cell_data(data=cp_stats, mesh=mesh.geometry)

    # OUTPUT 3: VTK cp_stats
    write_polydata(output_path / "cp_stats.vtp", polydata)
