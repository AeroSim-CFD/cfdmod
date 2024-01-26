import pathlib
from dataclasses import dataclass
from typing import Optional

import pandas as pd
from lnas import LnasFormat, LnasGeometry

from cfdmod.api.vtk.write_vtk import create_polydata_for_cell_data, merge_polydata, write_polydata
from cfdmod.use_cases.pressure.chunking import process_timestep_groups
from cfdmod.use_cases.pressure.extreme_values import ExtremeValuesParameters
from cfdmod.use_cases.pressure.geometry import (
    ProcessedEntity,
    get_excluded_entities,
    tabulate_geometry_data,
)
from cfdmod.use_cases.pressure.moment.Cm_config import CmConfig
from cfdmod.use_cases.pressure.moment.Cm_geom import (
    add_lever_arm_to_geometry_df,
    get_geometry_data,
    get_representative_volume,
)
from cfdmod.use_cases.pressure.path_manager import CmPathManager
from cfdmod.use_cases.pressure.zoning.body_config import BodyConfig
from cfdmod.use_cases.pressure.zoning.processing import (
    calculate_statistics,
    combine_stats_data_with_mesh,
)
from cfdmod.utils import create_folders_for_file


@dataclass
class CmOutputs:
    Cm_data: pd.DataFrame
    Cm_stats: pd.DataFrame
    Cm_regions: pd.DataFrame
    data_entity: ProcessedEntity
    excluded_entity: Optional[ProcessedEntity]

    def save_outputs(self, body_label: str, cfg_label: str, path_manager: CmPathManager):
        # Output 1: Cm_regions
        regions_path = path_manager.get_regions_df_path(body_label, cfg_label)
        create_folders_for_file(regions_path)
        self.Cm_regions.to_hdf(path_or_buf=regions_path, key="Regions", mode="w", index=False)

        # Output 2: Cm(t)
        timeseries_path = path_manager.get_timeseries_df_path(body_label, cfg_label)
        self.Cm_data.to_hdf(path_or_buf=timeseries_path, key="Cm_t", mode="w", index=False)

        # Output 3: Cm_stats
        stats_path = path_manager.get_stats_df_path(body_label, cfg_label)
        self.Cm_stats.to_hdf(path_or_buf=stats_path, key="Cm_stats", mode="w", index=False)

        # Output 4: VTK polydata
        all_entities = [self.data_entity]
        all_entities += [self.excluded_entity] if self.excluded_entity is not None else []
        merged_polydata = merge_polydata([entity.polydata for entity in all_entities])
        write_polydata(path_manager.get_vtp_path(body_label, cfg_label), merged_polydata)


def process_Cm(
    mesh: LnasFormat,
    body_cfg: BodyConfig,
    cfg: CmConfig,
    cp_path: pathlib.Path,
    extreme_params: ExtremeValuesParameters | None,
) -> CmOutputs:
    """Executes the moment coefficient processing routine

    Args:
        mesh (LnasFormat): Input mesh
        body_cfg (BodyConfig): Body configuration
        cfg (CmConfig): Moment coefficient configuration
        cp_path (pathlib.Path): Path for pressure coefficient time series
        extreme_params (ExtremeValuesParameters | None): Optional parameters for extreme values analysis

    Returns:
        CmOutputs: Compiled outputs for moment coefficient use case
    """
    geom_data = get_geometry_data(body_cfg=body_cfg, cfg=cfg, mesh=mesh)
    geometry_to_use = mesh.geometry.copy()
    geometry_to_use.apply_transformation(cfg.transformation.get_geometry_transformation())

    geometry_dict = {cfg.body: geom_data}
    geometry_df = tabulate_geometry_data(
        geom_dict=geometry_dict,
        mesh_areas=geometry_to_use.areas,
        mesh_normals=geometry_to_use.normals,
        transformation=cfg.transformation,
    )
    geometry_df = add_lever_arm_to_geometry_df(
        geom_data=geom_data,
        transformation=cfg.transformation,
        lever_origin=cfg.lever_origin,
        geometry_df=geometry_df,
    )
    Cm_data = process_timestep_groups(
        data_path=cp_path,
        geometry_df=geometry_df,
        geometry=geometry_to_use,
        processing_function=transform_Cm,
    )

    Cm_stats = calculate_statistics(
        historical_data=Cm_data,
        statistics_to_apply=cfg.statistics,
        variables=cfg.variables,
        group_by_key="region_idx",
        extreme_params=extreme_params,
    )

    body_data_df = combine_stats_data_with_mesh(
        mesh=geom_data.mesh,
        region_idx_array=geometry_df.region_idx.to_numpy(),
        data_stats=Cm_stats,
    )

    polydata = create_polydata_for_cell_data(body_data_df, geom_data.mesh)

    excluded_sfc_list = [sfc for sfc in mesh.surfaces.keys() if sfc not in body_cfg.surfaces]

    if len(excluded_sfc_list) != 0 and len(body_cfg.surfaces) != 0:
        excluded_entity = get_excluded_entities(
            excluded_sfc_list=excluded_sfc_list, mesh=mesh, data_columns=Cm_stats.columns
        )
    else:
        excluded_entity = None

    data_entity = ProcessedEntity(mesh=geom_data.mesh, polydata=polydata)

    cm_output = CmOutputs(
        Cm_data=Cm_data,
        Cm_stats=Cm_stats,
        Cm_regions=geometry_df,
        data_entity=data_entity,
        excluded_entity=excluded_entity,
    )

    return cm_output


def transform_Cm(
    raw_cp: pd.DataFrame, geometry_df: pd.DataFrame, geometry: LnasGeometry
) -> pd.DataFrame:
    """Transforms pressure coefficient into moment coefficient

    Args:
        raw_cp (pd.DataFrame): Body pressure coefficient data
        geometry_df (pd.DataFrame): Dataframe with geometric properties and triangle indexing
        geometry (LnasGeometry): Mesh geometry for bounding box definition

    Returns:
        pd.DataFrame: Moment coefficient dataframe
    """
    cp_data = pd.merge(raw_cp, geometry_df, on="point_idx", how="inner")
    cp_data["fx"] = -(cp_data["cp"] * cp_data["area"] * cp_data["n_x"])
    cp_data["fy"] = -(cp_data["cp"] * cp_data["area"] * cp_data["n_y"])
    cp_data["fz"] = -(cp_data["cp"] * cp_data["area"] * cp_data["n_z"])

    cp_data["mx"] = cp_data["ry"] * cp_data["fz"] - cp_data["rz"] * cp_data["fy"]  # y Fz - z Fy
    cp_data["my"] = cp_data["rz"] * cp_data["fx"] - cp_data["rx"] * cp_data["fz"]  # z Fx - x Fz
    cp_data["mz"] = cp_data["rx"] * cp_data["fy"] - cp_data["ry"] * cp_data["fx"]  # x Fy - y Fx

    Cm_data = (
        cp_data.groupby(["region_idx", "time_step"])  # type: ignore
        .agg(
            Mx=pd.NamedAgg(column="mx", aggfunc="sum"),
            My=pd.NamedAgg(column="my", aggfunc="sum"),
            Mz=pd.NamedAgg(column="mz", aggfunc="sum"),
        )
        .reset_index()
    )

    region_group_by = geometry_df.groupby(["region_idx"])
    representative_volume = {}

    for region_idx, region_points in region_group_by:
        region_points_idx = region_points.point_idx.to_numpy()
        V_rep = get_representative_volume(input_mesh=geometry, point_idx=region_points_idx)

        representative_volume[region_idx[0]] = {}
        representative_volume[region_idx[0]]["V_rep"] = V_rep

    rep_df = pd.DataFrame.from_dict(representative_volume, orient="index").reset_index()
    rep_df = rep_df.rename(columns={"index": "region_idx"})

    Cm_data = pd.merge(Cm_data, rep_df, on="region_idx")

    Cm_data["Cmx"] = Cm_data["Mx"] / Cm_data["V_rep"]
    Cm_data["Cmy"] = Cm_data["My"] / Cm_data["V_rep"]
    Cm_data["Cmz"] = Cm_data["Mz"] / Cm_data["V_rep"]

    Cm_data.drop(columns=["Mx", "My", "Mz", "V_rep"], inplace=True)

    return Cm_data
