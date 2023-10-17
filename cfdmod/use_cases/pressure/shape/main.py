import argparse
import pathlib
from dataclasses import dataclass

import numpy as np
import pandas as pd
from nassu.lnas import LagrangianFormat

from cfdmod.api.vtk.write_vtk import create_polydata_for_cell_data, merge_polydata, write_polydata
from cfdmod.logger import logger
from cfdmod.use_cases.pressure.path_manager import CePathManager
from cfdmod.use_cases.pressure.shape.Ce_config import CeConfig
from cfdmod.use_cases.pressure.shape.Ce_data import calculate_statistics, transform_to_Ce
from cfdmod.use_cases.pressure.shape.region_meshing import create_regions_mesh, get_mesh_bounds
from cfdmod.use_cases.pressure.shape.regions import get_region_index_mask
from cfdmod.use_cases.pressure.shape.zoning_config import ZoningModel


@dataclass
class ArgsModel:
    """Command line arguments for client app"""

    output: str
    cp: str
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
        "--cp",
        required=True,
        help="Path to body pressure coefficient series .hdf",
        type=str,
    )
    ap.add_argument(
        "--mesh",
        required=True,
        help="Path to LNAS normalized mesh file",
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
    path_manager = CePathManager(args_use.output, args_use.config, args_use.mesh, args_use.cp)
    post_proc_cfg = CeConfig.from_file(path_manager.config_path)

    logger.info("Reading mesh description...")
    mesh = LagrangianFormat.from_file(path_manager.mesh_path)
    logger.info("Mesh description loaded successfully!")

    logger.info("Preparing to read pressure coefficients data...")
    cp_data = pd.read_hdf(path_manager.cp_data_path)

    cp_data_to_use = cp_data.to_frame() if isinstance(cp_data, pd.Series) else cp_data
    logger.info("Read pressure coefficient data successfully!")

    n_timesteps = cp_data_to_use["time_step"].unique().shape[0]

    processed_polydata = []

    for cfg_label, cfg in post_proc_cfg.items():
        logger.info(f"Processing {cfg_label} ...")
        for sfc in mesh.surfaces.keys():
            if sfc in cfg.zoning.exclude:
                # Ignore surface
                logger.info(f"Surface {sfc} ignored!")
                continue

            sfc_mesh = mesh.geometry_from_surface(sfc)

            if sfc in cfg.zoning.no_zoning:
                bounds = get_mesh_bounds(sfc_mesh)
                zoning = ZoningModel(
                    x_intervals=[bounds[0][0], bounds[0][1]],
                    y_intervals=[bounds[1][0], bounds[1][1]],
                    z_intervals=[bounds[2][0], bounds[2][1]],
                )
            elif sfc in cfg.zoning.surfaces_in_exception:
                zoning = [cfg for cfg in cfg.zoning.exceptions.values() if sfc in cfg.surfaces][0]
            else:
                zoning = cfg.zoning.global_zoning

            zoning_to_use = zoning.offset_limits(0.1)

            logger.info(f"Processing surface {sfc} ...")
            # Output 1: Ce_regions
            df_regions = zoning_to_use.get_regions_df()

            df_regions.to_hdf(
                path_manager.get_regions_df_path(sfc, cfg_label),
                key="Regions",
                mode="w",
                index=False,
            )

            triangles_region = get_region_index_mask(mesh=sfc_mesh, df_regions=df_regions)

            sfc_triangles_idxs = mesh.surfaces[sfc].copy()

            # # Output 2: Ce(t)
            surface_ce = transform_to_Ce(
                surface_mesh=sfc_mesh,
                cp_data=cp_data_to_use,
                sfc_triangles_idxs=sfc_triangles_idxs,
                triangles_region=triangles_region,
                n_timesteps=n_timesteps,
            )

            surface_ce.to_hdf(
                path_manager.get_timeseries_df_path(sfc, cfg_label),
                key="Ce_t",
                mode="w",
                index=False,
            )

            # Output 3: Ce_stats
            surface_ce_stats = calculate_statistics(
                surface_ce, statistics_to_apply=post_proc_cfg[cfg_label].statistics
            )

            surface_ce_stats.to_hdf(
                path_manager.get_stats_df_path(sfc, cfg_label),
                key="Ce_stats",
                mode="w",
                index=False,
            )

            regions_mesh = create_regions_mesh(sfc_mesh, zoning_to_use)

            # Output 4: Regions Mesh
            regions_mesh.export_stl(
                path_manager.get_surface_path(sfc_label=sfc, cfg_label=cfg_label)
            )

            regions_mesh_triangles_region = get_region_index_mask(
                mesh=regions_mesh, df_regions=df_regions
            )

            region_data_df = pd.DataFrame()
            region_data_df["point_idx"] = np.arange(len(regions_mesh.triangle_vertices))
            region_data_df["region_idx"] = regions_mesh_triangles_region
            region_data_df = pd.merge(
                region_data_df, surface_ce_stats, on="region_idx", how="left"
            )
            region_data_df.drop(columns=["region_idx"], inplace=True)

            polydata = create_polydata_for_cell_data(region_data_df, regions_mesh)
            processed_polydata.append(polydata)

            logger.info(f"Processed surface {sfc}")

            if (region_data_df.isnull().sum() != 0).any():
                logger.warn(
                    "Region refinement is greater than data refinement. Resulted in NaN values"
                )
        merged_polydata = merge_polydata(processed_polydata)

        # Output 5: VTK
        write_polydata(path_manager.get_vtp_path(mesh.name, cfg_label), merged_polydata)

        logger.info(f"Merged polydata for {cfg_label}")
