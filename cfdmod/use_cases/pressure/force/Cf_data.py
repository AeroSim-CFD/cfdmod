import pathlib
from dataclasses import dataclass
from typing import Optional

import numpy as np
import pandas as pd
from lnas import LnasFormat, LnasGeometry

from cfdmod.api.vtk.write_vtk import create_polydata_for_cell_data, merge_polydata, write_polydata
from cfdmod.use_cases.pressure.chunking import process_timestep_groups
from cfdmod.use_cases.pressure.extreme_values import ExtremeValuesParameters
from cfdmod.use_cases.pressure.force.Cf_config import CfConfig
from cfdmod.use_cases.pressure.geometry import (
    GeometryData,
    ProcessedEntity,
    filter_geometry_from_list,
    get_excluded_entities,
    tabulate_geometry_data,
)
from cfdmod.use_cases.pressure.path_manager import CfPathManager
from cfdmod.use_cases.pressure.zoning.body_config import BodyConfig
from cfdmod.use_cases.pressure.zoning.processing import (
    calculate_statistics,
    combine_stats_data_with_mesh,
)
from cfdmod.utils import create_folders_for_file


@dataclass
class CfOutputs:
    Cf_data: pd.DataFrame
    Cf_stats: pd.DataFrame
    Cf_regions: pd.DataFrame
    data_entity: ProcessedEntity
    excluded_entity: Optional[ProcessedEntity]

    def save_outputs(self, body_label: str, cfg_label: str, path_manager: CfPathManager):
        # Output 1: Cf_regions
        regions_path = path_manager.get_regions_df_path(body_label, cfg_label)
        create_folders_for_file(regions_path)
        self.Cf_regions.to_hdf(regions_path, key="Regions", mode="w", index=False)

        # Output 2: Cf(t)
        timeseries_path = path_manager.get_timeseries_df_path(body_label, cfg_label)
        self.Cf_data.to_hdf(timeseries_path, key="Cf_t", mode="w", index=False)

        # Output 3: Cf_stats
        stats_path = path_manager.get_stats_df_path(body_label, cfg_label)
        self.Cf_stats.to_hdf(stats_path, key="Cf_stats", mode="w", index=False)

        # Output 4: VTK polydata
        all_entities = [self.data_entity]
        all_entities += [self.excluded_entity] if self.excluded_entity is not None else []
        merged_polydata = merge_polydata([entity.polydata for entity in all_entities])
        write_polydata(path_manager.get_vtp_path(body_label, cfg_label), merged_polydata)


def get_geometry_data(body_cfg: BodyConfig, cfg: CfConfig, mesh: LnasFormat) -> GeometryData:
    """Builds a GeometryData from the mesh and the configurations

    Args:
        body_cfg (BodyConfig): Body configuration with surface list
        cfg (CfConfig): Force coefficient configuration
        mesh (LnasFormat): Input mesh

    Returns:
        GeometryData: Filtered GeometryData
    """
    if len(body_cfg.surfaces) == 0:
        # Include all surfaces
        geometry_idx = np.arange(0, len(mesh.geometry.triangles))
        geom = mesh.geometry
    else:
        # Filter mesh for all surfaces
        geom, geometry_idx = filter_geometry_from_list(mesh=mesh, sfc_list=body_cfg.surfaces)

    return GeometryData(mesh=geom, zoning_to_use=cfg.sub_bodies, triangles_idxs=geometry_idx)


def process_Cf(
    mesh: LnasFormat,
    body_cfg: BodyConfig,
    cfg: CfConfig,
    cp_path: pathlib.Path,
    extreme_params: ExtremeValuesParameters | None,
) -> CfOutputs:
    """Executes the force coefficient processing routine

    Args:
        mesh (LnasFormat): Input mesh
        body_cfg (BodyConfig): Body configuration
        cfg (CfConfig): Force coefficient configuration
        cp_path (pathlib.Path): Path for pressure coefficient time series
        extreme_params (ExtremeValuesParameters | None): Optional parameters for extreme values analysis

    Returns:
        CfOutputs: Compiled outputs for force coefficient use case
    """
    geom_data = get_geometry_data(body_cfg=body_cfg, cfg=cfg, mesh=mesh)
    geometry_dict = {cfg.body: geom_data}
    geometry_df = tabulate_geometry_data(
        geom_dict=geometry_dict,
        mesh_areas=mesh.geometry.areas,
        mesh_normals=mesh.geometry.normals,
        transformation=cfg.transformation,
    )
    Cf_data = process_timestep_groups(
        data_path=cp_path,
        geometry_df=geometry_df,
        geometry=mesh.geometry,
        processing_function=transform_Cf,
    )

    Cf_stats = calculate_statistics(
        historical_data=Cf_data,
        statistics_to_apply=cfg.statistics,
        variables=cfg.variables,
        group_by_key="region_idx",
        extreme_params=extreme_params,
    )

    body_data_df = combine_stats_data_with_mesh(
        mesh=geom_data.mesh,
        region_idx_array=geometry_df.region_idx.to_numpy(),
        data_stats=Cf_stats,
    )

    polydata = create_polydata_for_cell_data(body_data_df, geom_data.mesh)

    excluded_sfc_list = [sfc for sfc in mesh.surfaces.keys() if sfc not in body_cfg.surfaces]

    if len(excluded_sfc_list) != 0:
        excluded_entity = get_excluded_entities(
            excluded_sfc_list=excluded_sfc_list, mesh=mesh, data_columns=Cf_stats.columns
        )
    else:
        excluded_entity = None

    data_entity = ProcessedEntity(mesh=geom_data.mesh, polydata=polydata)

    cf_output = CfOutputs(
        Cf_data=Cf_data,
        Cf_stats=Cf_stats,
        Cf_regions=geometry_df,
        data_entity=data_entity,
        excluded_entity=excluded_entity,
    )

    return cf_output


def transform_Cf(
    raw_cp: pd.DataFrame, geometry_df: pd.DataFrame, geometry: LnasGeometry
) -> pd.DataFrame:
    """Transforms pressure coefficient into force coefficient

    Args:
        raw_cp (pd.DataFrame): Body pressure coefficient data
        geometry_df (pd.DataFrame): Dataframe with geometric properties and triangle indexing
        geometry (LnasGeometry): Mesh geometry for bounding box definition

    Returns:
        pd.DataFrame: Force coefficient dataframe
    """
    cp_data = pd.merge(raw_cp, geometry_df, on="point_idx", how="inner")
    cp_data["fx"] = -(cp_data["cp"] * cp_data["area"] * cp_data["n_x"])
    cp_data["fy"] = -(cp_data["cp"] * cp_data["area"] * cp_data["n_y"])
    cp_data["fz"] = -(cp_data["cp"] * cp_data["area"] * cp_data["n_z"])

    Cf_data = (
        cp_data.groupby(["region_idx", "time_step"])  # type: ignore
        .agg(
            Fx=pd.NamedAgg(column="fx", aggfunc="sum"),
            Fy=pd.NamedAgg(column="fy", aggfunc="sum"),
            Fz=pd.NamedAgg(column="fz", aggfunc="sum"),
        )
        .reset_index()
    )

    region_group_by = geometry_df.groupby(["region_idx"])
    representative_areas = {}

    for region_idx, region_points in region_group_by:
        region_points_idx = region_points.point_idx.to_numpy()
        Ax, Ay, Az = get_representative_areas(input_mesh=geometry, point_idx=region_points_idx)

        representative_areas[region_idx[0]] = {}
        representative_areas[region_idx[0]]["ATx"] = Ax
        representative_areas[region_idx[0]]["ATy"] = Ay
        representative_areas[region_idx[0]]["ATz"] = Az

    rep_df = pd.DataFrame.from_dict(representative_areas, orient="index").reset_index()
    rep_df = rep_df.rename(columns={"index": "region_idx"})
    Cf_data = pd.merge(Cf_data, rep_df, on="region_idx")

    Cf_data["Cfx"] = Cf_data["Fx"] / Cf_data["ATx"]
    Cf_data["Cfy"] = Cf_data["Fy"] / Cf_data["ATy"]
    Cf_data["Cfz"] = Cf_data["Fz"] / Cf_data["ATz"]

    Cf_data.drop(columns=["Fx", "Fy", "Fz", "ATx", "ATy", "ATz"], inplace=True)

    return Cf_data


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

    # Threshold to avoid big coefficients
    Lx = 1 if Lx < 1 else Lx
    Ly = 1 if Ly < 1 else Ly
    Lz = 1 if Lz < 1 else Lz

    Ax = Ly * Lz
    Ay = Lx * Lz
    Az = Lx * Ly

    return Ax, Ay, Az
