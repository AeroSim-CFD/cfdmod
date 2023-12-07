from dataclasses import dataclass

import numpy as np
import pandas as pd
from lnas import LnasFormat, LnasGeometry
from vtk import vtkAppendPolyData, vtkPolyData

from cfdmod.api.vtk.write_vtk import create_polydata_for_cell_data, write_polydata
from cfdmod.use_cases.pressure.geometry import get_geometry_from_mesh
from cfdmod.use_cases.pressure.moment.Cm_config import CmConfig
from cfdmod.use_cases.pressure.path_manager import CmPathManager
from cfdmod.use_cases.pressure.zoning.body_config import BodyConfig
from cfdmod.use_cases.pressure.zoning.processing import (
    calculate_statistics,
    combine_stats_data_with_mesh,
    get_indexing_mask,
)
from cfdmod.utils import create_folders_for_file


@dataclass
class ProcessedBodyData:
    df_regions: pd.DataFrame
    body_cm: pd.DataFrame
    body_cm_stats: pd.DataFrame
    body_geom: LnasGeometry
    body_data_df: pd.DataFrame
    polydata: vtkPolyData | vtkAppendPolyData

    def save_outputs(self, body_label: str, cfg_label: str, path_manager: CmPathManager):
        # Output 1: Cm(t)
        timeseries_path = path_manager.get_timeseries_df_path(
            body_label=body_label, cfg_label=cfg_label
        )
        create_folders_for_file(timeseries_path)
        self.body_cm.to_hdf(timeseries_path, key="Cm_t", mode="w", index=False)

        # Output 2: Cm_stats
        stats_path = path_manager.get_stats_df_path(body_label=body_label, cfg_label=cfg_label)
        create_folders_for_file(stats_path)
        self.body_cm_stats.to_hdf(stats_path, key="Cm_stats", mode="w", index=False)

        # Output 3: VTK
        vtp_path = path_manager.get_vtp_path(body_label=body_label, cfg_label=cfg_label)
        create_folders_for_file(vtp_path)
        write_polydata(output_filename=vtp_path, poly_data=self.polydata)


def process_body(
    mesh: LnasFormat, body_cfg: BodyConfig, cp_data: pd.DataFrame, cfg: CmConfig
) -> ProcessedBodyData:
    """Processes a sub body from separating the surfaces of the original mesh
    The pressure coefficient must already contain the areas of each triangle.
    It must be added before calling this function

    Args:
        mesh (LnasFormat): LNAS mesh
        body_cfg (BodyConfig): Body processing configuration
        cp_data (pd.DataFrame): Pressure coefficients data
        cfg (CmConfig): Post processing configuration

    Returns:
        ProcessedBodyData: Processed body object
    """
    body_geom, geometry_idx = get_geometry_from_mesh(body_cfg=body_cfg, mesh=mesh)

    zoning_to_use = cfg.sub_bodies.offset_limits(0.1)
    df_regions = zoning_to_use.get_regions_df()

    transformed_body = body_geom.copy()
    transformed_body.apply_transformation(cfg.transformation.get_geometry_transformation())

    sub_body_idx_array = get_indexing_mask(transformed_body, df_regions)
    # sub_body_idx_array = get_indexing_mask(body_geom, df_regions)
    sub_body_idx = pd.DataFrame({"point_idx": geometry_idx, "region_idx": sub_body_idx_array})

    body_data = cp_data[cp_data["point_idx"].isin(geometry_idx)].copy()
    body_data = pd.merge(body_data, sub_body_idx, on="point_idx", how="left")

    centroids = np.mean(transformed_body.triangle_vertices, axis=1)
    # centroids = np.mean(body_geom.triangle_vertices, axis=1)

    position_df = get_lever_relative_position_df(
        centroids=centroids, lever_origin=cfg.lever_origin, geometry_idx=geometry_idx
    )
    body_data = pd.merge(body_data, position_df, on="point_idx", how="left")

    body_cf = transform_to_Cm(body_data=body_data, body_geom=body_geom)

    body_cf_stats = calculate_statistics(body_cf, cfg.statistics, variables=cfg.variables)

    body_data_df = combine_stats_data_with_mesh(
        mesh=body_geom, region_idx_array=sub_body_idx_array, data_stats=body_cf_stats
    )

    polydata = create_polydata_for_cell_data(body_data_df, body_geom)

    return ProcessedBodyData(
        df_regions=df_regions,
        body_cm=body_cf,
        body_cm_stats=body_cf_stats,
        body_geom=body_geom,
        body_data_df=body_data_df,
        polydata=polydata,
    )


def transform_to_Cm(body_data: pd.DataFrame, body_geom: LnasGeometry) -> pd.DataFrame:
    """Converts pressure coefficient data for a body into force coefficients
    The pressure coefficient must already contain the areas of each triangle.
    It must be added before calling this function

    Args:
        body_data (pd.DataFrame): Pressure coefficient data for the body
        body_geom (LnasGeometry): LNAS mesh for the body geometry

    Returns:
        pd.DataFrame: Body force coefficients data
    """
    body_data["fx"] = body_data["cp"] * body_data["Ax"]
    body_data["fy"] = body_data["cp"] * body_data["Ay"]
    body_data["fz"] = body_data["cp"] * body_data["Az"]

    body_data["mx"] = (
        body_data["ry"] * body_data["fz"] - body_data["rz"] * body_data["fy"]
    )  # y Fz - z Fy
    body_data["my"] = (
        body_data["rz"] * body_data["fx"] - body_data["rx"] * body_data["fz"]
    )  # z Fx - x Fz
    body_data["mz"] = (
        body_data["rx"] * body_data["fy"] - body_data["ry"] * body_data["fx"]
    )  # x Fy - y Fx

    body_cm = (
        body_data.groupby(["region_idx", "time_step"])  # type: ignore
        .agg(
            Mx=pd.NamedAgg(column="mx", aggfunc="sum"),
            My=pd.NamedAgg(column="my", aggfunc="sum"),
            Mz=pd.NamedAgg(column="mz", aggfunc="sum"),
        )
        .reset_index()
    )

    V_rep = get_representative_volume(body_geom)

    body_cm["Cmx"] = body_cm["Mx"] / V_rep
    body_cm["Cmy"] = body_cm["My"] / V_rep
    body_cm["Cmz"] = body_cm["Mz"] / V_rep

    return body_cm


def get_lever_relative_position_df(
    centroids: np.ndarray, lever_origin: tuple[float, float, float], geometry_idx: np.ndarray
) -> pd.DataFrame:
    """Creates a Dataframe with the relative position for each triangle

    Args:
        centroids (np.ndarray): Array of triangle centroids
        lever_origin (tuple[float, float, float]): Coordinate of the lever origin
        geometry_idx (np.ndarray): Indexes of the triangles of the vody

    Returns:
        pd.DataFrame: Relative position dataframe
    """
    position_df = pd.DataFrame()
    position_df["rx"] = centroids[:, 0] - lever_origin[0]
    position_df["ry"] = centroids[:, 1] - lever_origin[1]
    position_df["rz"] = centroids[:, 2] - lever_origin[2]
    position_df["point_idx"] = geometry_idx

    return position_df


def get_representative_volume(input_mesh: LnasGeometry) -> float:
    """Calculates the representative volume from the bounding box of a given mesh

    Args:
        input_mesh (LnasGeometry): Input LNAS mesh

    Returns:
        float: Representative volume value
    """
    geom_verts = input_mesh.triangle_vertices.reshape(-1, 3)
    x_min, x_max = geom_verts[:, 0].min(), geom_verts[:, 0].max()
    y_min, y_max = geom_verts[:, 1].min(), geom_verts[:, 1].max()
    z_min, z_max = geom_verts[:, 2].min(), geom_verts[:, 2].max()

    Lx = x_max - x_min
    Ly = y_max - y_min
    Lz = z_max - z_min

    return Lx * Ly * Lz
