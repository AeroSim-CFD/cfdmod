from dataclasses import dataclass

import numpy as np
import pandas as pd
from nassu.lnas import LagrangianFormat, LagrangianGeometry
from vtk import vtkPolyData

from cfdmod.api.vtk.write_vtk import create_polydata_for_cell_data, write_polydata
from cfdmod.use_cases.pressure.force.Cf_config import CfConfig
from cfdmod.use_cases.pressure.path_manager import CfPathManager
from cfdmod.use_cases.pressure.zoning.body_config import BodyConfig
from cfdmod.use_cases.pressure.zoning.processing import (
    calculate_statistics,
    combine_stats_data_with_mesh,
    get_indexing_mask,
)


@dataclass
class ProcessedBodyData:
    df_regions: pd.DataFrame
    body_cf: pd.DataFrame
    body_cf_stats: pd.DataFrame
    body_geom: LagrangianGeometry
    body_data_df: pd.DataFrame
    polydata: vtkPolyData

    def save_outputs(self, body_label: str, cfg_label: str, path_manager: CfPathManager):
        # Output 1: Ce(t)
        self.body_cf.to_hdf(
            path_manager.get_timeseries_df_path(body_label=body_label, cfg_label=cfg_label),
            key="Ce_t",
            mode="w",
            index=False,
        )

        # Output 2: Ce_stats
        self.body_cf_stats.to_hdf(
            path_manager.get_stats_df_path(body_label=body_label, cfg_label=cfg_label),
            key="Ce_stats",
            mode="w",
            index=False,
        )

        # Output 3: VTK
        write_polydata(
            path_manager.get_vtp_path(body_label=body_label, cfg_label=cfg_label),
            self.polydata,
        )


def process_body(
    mesh: LagrangianFormat, body_cfg: BodyConfig, cp_data: pd.DataFrame, cfg: CfConfig
) -> ProcessedBodyData:
    """Processes a sub body from separating the surfaces of the original mesh
    The pressure coefficient must already contain the areas of each triangle.
    It must be added before calling this function

    Args:
        mesh (LagrangianFormat): LNAS mesh
        body_cfg (BodyConfig): Body processing configuration
        cp_data (pd.DataFrame): Pressure coefficients data
        cfg (CfConfig): Post processing configuration

    Returns:
        ProcessedBodyData: Processed body object
    """
    body_geom, geometry_idx = get_geometry_from_mesh(body_cfg=body_cfg, mesh=mesh)

    zoning_to_use = body_cfg.sub_bodies.offset_limits(0.1)
    df_regions = zoning_to_use.get_regions_df()

    sub_body_idx_array = get_indexing_mask(body_geom, df_regions)
    sub_body_idx = pd.DataFrame({"point_idx": geometry_idx, "region_idx": sub_body_idx_array})

    body_data = cp_data[cp_data["point_idx"].isin(geometry_idx)].copy()
    body_data = pd.merge(body_data, sub_body_idx, on="point_idx", how="left")

    body_cf = transform_to_Cf(body_data=body_data, body_geom=body_geom)

    body_cf_stats = calculate_statistics(body_cf, cfg.statistics, variables=cfg.variables)

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


def transform_to_Cf(body_data: pd.DataFrame, body_geom: LagrangianGeometry) -> pd.DataFrame:
    """Converts pressure coefficient data for a body into force coefficients
    The pressure coefficient must already contain the areas of each triangle.
    It must be added before calling this function

    Args:
        body_data (pd.DataFrame): Pressure coefficient data for the body
        body_geom (LagrangianGeometry): LNAS mesh for the body geometry

    Returns:
        pd.DataFrame: Body force coefficients data
    """
    body_data["fx"] = body_data["cp"] * body_data["Ax"]
    body_data["fy"] = body_data["cp"] * body_data["Ay"]
    body_data["fz"] = body_data["cp"] * body_data["Az"]

    body_cf = (
        body_data.groupby(["region_idx", "time_step"])  # type: ignore
        .agg(
            Fx=pd.NamedAgg(column="fx", aggfunc="sum"),
            Fy=pd.NamedAgg(column="fy", aggfunc="sum"),
            Fz=pd.NamedAgg(column="fz", aggfunc="sum"),
        )
        .reset_index()
    )

    Ax, Ay, Az = get_representative_areas(body_geom)

    body_cf["Cfx"] = body_cf["Fx"] / Ax
    body_cf["Cfy"] = body_cf["Fy"] / Ay
    body_cf["Cfz"] = body_cf["Fz"] / Az

    return body_cf


def get_geometry_from_mesh(
    body_cfg: BodyConfig, mesh: LagrangianFormat
) -> tuple[LagrangianGeometry, np.ndarray]:
    """Filters the mesh from the list of surfaces that define the body in config

    Args:
        body_cfg (BodyConfig): Body configuration
        mesh (LagrangianFormat): LNAS mesh

    Raises:
        Exception: Surface specified is not defined in LNAS

    Returns:
        tuple[LagrangianGeometry, np.ndarray]: Tuple containing the body geometry and the filtered triangle indexes
    """
    if len(body_cfg.surfaces) == 0:
        # Include all surfaces
        geometry_idx = np.arange(0, len(mesh.geometry.triangles))
    else:
        # Filter mesh for all surfaces
        geometry_idx = np.array([], dtype=np.int32)
        for sfc in body_cfg.surfaces:
            if sfc not in mesh.surfaces.keys():
                raise Exception("Surface defined in body is not separated in the LNAS file.")
            geometry_idx = np.concatenate((geometry_idx, mesh.surfaces[sfc]))

    body_geom = LagrangianGeometry(
        vertices=mesh.geometry.vertices.copy(),
        triangles=mesh.geometry.triangles[geometry_idx].copy(),
    )

    return body_geom, geometry_idx


def get_representative_areas(input_mesh: LagrangianGeometry) -> tuple[float, float, float]:
    """Calculates the representative areas from the bounding box of a given mesh

    Args:
        input_mesh (LagrangianGeometry): Input LNAS mesh

    Returns:
        tuple[float, float, float]: Representative areas tuple (Ax, Ay, Az)
    """
    x_min, x_max = input_mesh.vertices[:, 0].min(), input_mesh.vertices[:, 0].max()
    y_min, y_max = input_mesh.vertices[:, 1].min(), input_mesh.vertices[:, 1].max()
    z_min, z_max = input_mesh.vertices[:, 2].min(), input_mesh.vertices[:, 2].max()

    Lx = x_max - x_min
    Ly = y_max - y_min
    Lz = z_max - z_min

    Ax = Ly * Lz
    Ay = Lx * Lz
    Az = Lx * Ly

    return Ax, Ay, Az
