from dataclasses import dataclass
from typing import Optional

import numpy as np
import pandas as pd
from lnas import LnasFormat, LnasGeometry
from vtk import vtkAppendPolyData, vtkPolyData

from cfdmod.api.vtk.write_vtk import create_polydata_for_cell_data, write_polydata
from cfdmod.use_cases.pressure.extreme_values import ExtremeValuesParameters
from cfdmod.use_cases.pressure.force.Cf_config import CfConfig
from cfdmod.use_cases.pressure.geometry import get_geometry_from_mesh
from cfdmod.use_cases.pressure.path_manager import CfPathManager
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
    body_cf: pd.DataFrame
    body_cf_stats: pd.DataFrame
    body_geom: LnasGeometry
    body_data_df: pd.DataFrame
    polydata: vtkPolyData | vtkAppendPolyData

    def save_outputs(self, body_label: str, cfg_label: str, path_manager: CfPathManager):
        # Output 1: Cf(t)
        timeseries_path = path_manager.get_timeseries_df_path(
            body_label=body_label, cfg_label=cfg_label
        )
        create_folders_for_file(timeseries_path)
        self.body_cf.to_hdf(timeseries_path, key="Cf_t", mode="w", index=False)

        # Output 2: Cf_stats
        stats_path = path_manager.get_stats_df_path(body_label=body_label, cfg_label=cfg_label)
        create_folders_for_file(stats_path)
        self.body_cf_stats.to_hdf(stats_path, key="Cf_stats", mode="w", index=False)

        # Output 3: VTK
        vtp_path = path_manager.get_vtp_path(body_label=body_label, cfg_label=cfg_label)
        create_folders_for_file(vtp_path)
        write_polydata(vtp_path, self.polydata)


def process_body(
    mesh: LnasFormat,
    body_cfg: BodyConfig,
    cp_data: pd.DataFrame,
    cfg: CfConfig,
    extreme_params: Optional[ExtremeValuesParameters] = None,
) -> ProcessedBodyData:
    """Processes a sub body from separating the surfaces of the original mesh
    The pressure coefficient must already contain the areas of each triangle.
    It must be added before calling this function

    Args:
        mesh (LnasFormat): LNAS mesh
        body_cfg (BodyConfig): Body processing configuration
        cp_data (pd.DataFrame): Pressure coefficients data
        cfg (CfConfig): Post processing configuration
        extreme_params (Optional[ExtremeValuesParameters]): Parameters for extreme values analysis. Defaults to None.

    Returns:
        ProcessedBodyData: Processed body object
    """
    body_geom, geometry_idx = get_geometry_from_mesh(body_cfg=body_cfg, mesh=mesh)

    zoning_to_use = cfg.sub_bodies.offset_limits(0.1)
    df_regions = zoning_to_use.get_regions_df()

    transformed_body = body_geom.copy()
    transformed_body.apply_transformation(cfg.transformation.get_geometry_transformation())

    sub_body_idx_array = get_indexing_mask(transformed_body, df_regions)
    sub_body_idx = pd.DataFrame({"point_idx": geometry_idx, "region_idx": sub_body_idx_array})

    mask = cp_data["point_idx"].isin(geometry_idx)
    body_data = cp_data[mask].copy()
    body_data = pd.merge(body_data, sub_body_idx, on="point_idx", how="left")

    body_cf = transform_to_Cf(
        body_data=body_data, sub_body_idx_df=sub_body_idx, body_geom=mesh.geometry
    )

    body_cf_stats = calculate_statistics(
        body_cf,
        cfg.statistics,
        variables=cfg.variables,
        group_by_key="region_idx",
        extreme_params=extreme_params,
    )

    body_data_df = combine_stats_data_with_mesh(
        mesh=body_geom, region_idx_array=sub_body_idx_array, data_stats=body_cf_stats
    )

    polydata = create_polydata_for_cell_data(body_data_df, body_geom)

    return ProcessedBodyData(
        df_regions=df_regions,
        body_cf=body_cf,
        body_cf_stats=body_cf_stats,
        body_geom=body_geom,
        body_data_df=body_data_df,
        polydata=polydata,
    )


def transform_to_Cf(
    body_data: pd.DataFrame, sub_body_idx_df: pd.DataFrame, body_geom: LnasGeometry
) -> pd.DataFrame:
    """Converts pressure coefficient data for a body into force coefficients
    The pressure coefficient must already contain the areas of each triangle.
    It must be added before calling this function

    Args:
        body_data (pd.DataFrame): Pressure coefficient data for the body
        sub_body_idx_df (pd.DataFrame): Dataframe grouping point index for each region index
        body_geom (LnasGeometry): LNAS mesh for the body geometry

    Returns:
        pd.DataFrame: Body force coefficients data
    """

    def sum_positive_values(series):
        # TODO: Refactor representative area method
        return series[series > 0].sum()

    body_data["fx"] = -(body_data["cp"] * body_data["Ax"])
    body_data["fy"] = -(body_data["cp"] * body_data["Ay"])
    body_data["fz"] = -(body_data["cp"] * body_data["Az"])

    body_cf = (
        body_data.groupby(["region_idx", "time_step"])  # type: ignore
        .agg(
            Fx=pd.NamedAgg(column="fx", aggfunc="sum"),
            Fy=pd.NamedAgg(column="fy", aggfunc="sum"),
            Fz=pd.NamedAgg(column="fz", aggfunc="sum"),
        )
        .reset_index()
    )

    region_group_by = sub_body_idx_df.groupby(["region_idx"])
    representative_areas = {}

    for region_idx, region_points in region_group_by:
        region_points_idx = region_points.point_idx.to_numpy()
        Ax, Ay, Az = get_representative_areas(input_mesh=body_geom, point_idx=region_points_idx)

        representative_areas[region_idx[0]] = {}
        representative_areas[region_idx[0]]["ATx"] = Ax
        representative_areas[region_idx[0]]["ATy"] = Ay
        representative_areas[region_idx[0]]["ATz"] = Az

    rep_df = pd.DataFrame.from_dict(representative_areas, orient="index").reset_index()
    rep_df = rep_df.rename(columns={"index": "region_idx"})
    body_cf = pd.merge(body_cf, rep_df, on="region_idx")

    body_cf["Cfx"] = body_cf["Fx"] / body_cf["ATx"]
    body_cf["Cfy"] = body_cf["Fy"] / body_cf["ATy"]
    body_cf["Cfz"] = body_cf["Fz"] / body_cf["ATz"]

    return body_cf


def get_representative_areas(
    input_mesh: LnasGeometry, point_idx: np.ndarray
) -> tuple[float, float, float]:
    """Calculates the representative areas from the bounding box of a given mesh

    Args:
        input_mesh (LnasGeometry): Input LNAS mesh
        point_idx (np.ndarray): Array of triangle indices of each sub region

    Returns:
        tuple[float, float, float]: Representative areas tuple (Ax, Ay, Az)
    """
    geom_verts = input_mesh.triangle_vertices[point_idx].reshape(-1, 3)
    x_min, x_max = geom_verts[:, 0].min(), geom_verts[:, 0].max()
    y_min, y_max = geom_verts[:, 1].min(), geom_verts[:, 1].max()
    z_min, z_max = geom_verts[:, 2].min(), geom_verts[:, 2].max()

    Lx = x_max - x_min
    Ly = y_max - y_min
    Lz = z_max - z_min

    Ax = Ly * Lz
    Ay = Lx * Lz
    Az = Lx * Ly

    return Ax, Ay, Az
