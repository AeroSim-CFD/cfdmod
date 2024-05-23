import pathlib

import pandas as pd
from lnas import LnasFormat, LnasGeometry

from cfdmod.api.vtk.write_vtk import create_polydata_for_cell_data
from cfdmod.use_cases.pressure.chunking import process_timestep_groups
from cfdmod.use_cases.pressure.geometry import (
    GeometryData,
    ProcessedEntity,
    get_excluded_entities,
    get_geometry_data,
    get_region_definition_dataframe,
    tabulate_geometry_data,
)
from cfdmod.use_cases.pressure.moment.Cm_config import CmConfig
from cfdmod.use_cases.pressure.moment.Cm_geom import (
    add_lever_arm_to_geometry_df,
    get_representative_volume,
)
from cfdmod.use_cases.pressure.output import CommonOutput
from cfdmod.use_cases.pressure.zoning.body_config import BodyDefinition
from cfdmod.use_cases.pressure.zoning.processing import (
    calculate_statistics,
    combine_stats_data_with_mesh,
)
from cfdmod.utils import convert_dataframe_into_matrix


def process_Cm(
    mesh: LnasFormat,
    cfg: CmConfig,
    cp_path: pathlib.Path,
    bodies_definition: dict[str, BodyDefinition],
) -> dict[str, CommonOutput]:
    """Executes the moment coefficient processing routine

    Args:
        mesh (LnasFormat): Input mesh
        cfg (CmConfig): Moment coefficient configuration
        cp_path (pathlib.Path): Path for pressure coefficient time series
        bodies_definition (dict[str, BodyDefinition]): Dictionary of bodies definition

    Returns:
        dict[str, CommonOutput]: Compiled outputs for moment coefficient use case keyed by direction
    """
    geometry_dict: dict[str, GeometryData] = {}
    for body_cfg in cfg.bodies:
        geom_data = get_geometry_data(
            body_cfg=body_cfg, sfc_list=bodies_definition[body_cfg.name].surfaces, mesh=mesh
        )
        geometry_dict[body_cfg.name] = geom_data

    geometry_to_use = mesh.geometry.copy()
    geometry_to_use.apply_transformation(cfg.transformation.get_geometry_transformation())
    geometry_df = tabulate_geometry_data(
        geom_dict=geometry_dict,
        mesh_areas=geometry_to_use.areas,
        mesh_normals=geometry_to_use.normals,
        transformation=cfg.transformation,
    )
    for body_cfg in cfg.bodies:
        geometry_df = add_lever_arm_to_geometry_df(
            geom_data=geometry_dict[body_cfg.name],
            transformation=cfg.transformation,
            lever_origin=body_cfg.lever_origin,
            geometry_df=geometry_df,
        )

    Cm_data = process_timestep_groups(
        data_path=cp_path,
        geometry_df=geometry_df,
        geometry=geometry_to_use,
        processing_function=transform_Cm,
    )

    included_sfc_list = [
        sfc for body_cfg in cfg.bodies for sfc in bodies_definition[body_cfg.name].surfaces
    ]
    excluded_sfc_list = [sfc for sfc in mesh.surfaces.keys() if sfc not in included_sfc_list]

    if len(excluded_sfc_list) != 0:
        col = [s.stats for s in cfg.statistics]
        excluded_entity = [
            get_excluded_entities(excluded_sfc_list=excluded_sfc_list, mesh=mesh, data_columns=col)
        ]
    else:
        excluded_entity = []

    compild_cm_output = {}
    for direction_lbl in cfg.directions:
        Cm_dir_data = convert_dataframe_into_matrix(
            Cm_data[["region_idx", "time_normalized", f"Cm{direction_lbl}"]],
            row_data_label="time_normalized",
            column_data_label="region_idx",
            value_data_label=f"Cm{direction_lbl}",
        )
        Cm_stats = calculate_statistics(
            historical_data=Cm_dir_data, statistics_to_apply=cfg.statistics
        )

        processed_entities: list[ProcessedEntity] = []
        for body_cfg in cfg.bodies:
            body_data = geometry_dict[body_cfg.name]
            region_idx_arr = geometry_df.loc[
                geometry_df.region_idx.str.contains(body_cfg.name)
            ].region_idx.to_numpy()

            body_data_df = combine_stats_data_with_mesh(
                mesh=body_data.mesh,
                region_idx_array=region_idx_arr,
                data_stats=Cm_stats,
            )

            polydata = create_polydata_for_cell_data(body_data_df, body_data.mesh)
            data_entity = ProcessedEntity(mesh=body_data.mesh, polydata=polydata)
            processed_entities.append(data_entity)

        compild_cm_output[direction_lbl] = CommonOutput(
            data_df=Cm_dir_data,
            stats_df=Cm_stats,
            processed_entities=processed_entities,
            excluded_entities=excluded_entity,
            region_indexing_df=geometry_df[["region_idx", "point_idx"]],
            region_definition_df=get_region_definition_dataframe(geometry_dict),
        )

    return compild_cm_output


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
        cp_data.groupby(["region_idx", "time_normalized"])  # type: ignore
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
