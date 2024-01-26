import pathlib

import pandas as pd
from lnas import LnasFormat, LnasGeometry

from cfdmod.api.vtk.write_vtk import create_polydata_for_cell_data
from cfdmod.use_cases.pressure.chunking import process_timestep_groups
from cfdmod.use_cases.pressure.extreme_values import ExtremeValuesParameters
from cfdmod.use_cases.pressure.force.Cf_config import CfConfig
from cfdmod.use_cases.pressure.force.Cf_geom import get_geometry_data, get_representative_areas
from cfdmod.use_cases.pressure.geometry import (
    ProcessedEntity,
    get_excluded_entities,
    tabulate_geometry_data,
)
from cfdmod.use_cases.pressure.output import CommonOutput
from cfdmod.use_cases.pressure.zoning.body_config import BodyConfig
from cfdmod.use_cases.pressure.zoning.processing import (
    calculate_statistics,
    combine_stats_data_with_mesh,
)


def process_Cf(
    mesh: LnasFormat,
    body_cfg: BodyConfig,
    cfg: CfConfig,
    cp_path: pathlib.Path,
    extreme_params: ExtremeValuesParameters | None,
) -> CommonOutput:
    """Executes the force coefficient processing routine

    Args:
        mesh (LnasFormat): Input mesh
        body_cfg (BodyConfig): Body configuration
        cfg (CfConfig): Force coefficient configuration
        cp_path (pathlib.Path): Path for pressure coefficient time series
        extreme_params (ExtremeValuesParameters | None): Optional parameters for extreme values analysis

    Returns:
        CommonOutput: Compiled outputs for force coefficient use case
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
    Cf_data = process_timestep_groups(
        data_path=cp_path,
        geometry_df=geometry_df,
        geometry=geometry_to_use,
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
        col = Cf_stats.columns
        excluded_entity = [
            get_excluded_entities(excluded_sfc_list=excluded_sfc_list, mesh=mesh, data_columns=col)
        ]
    else:
        excluded_entity = []

    data_entity = ProcessedEntity(mesh=geom_data.mesh, polydata=polydata)

    cf_output = CommonOutput(
        data_df=Cf_data,
        stats_df=Cf_stats,
        regions_df=geometry_df,
        processed_entities=[data_entity],
        excluded_entities=excluded_entity,
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
