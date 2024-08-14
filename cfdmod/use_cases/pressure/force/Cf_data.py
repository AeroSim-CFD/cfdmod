import pathlib

import numpy as np
import pandas as pd
from lnas import LnasFormat, LnasGeometry

from cfdmod.api.vtk.write_vtk import create_polydata_for_cell_data
from cfdmod.use_cases.pressure.chunking import process_timestep_groups
from cfdmod.use_cases.pressure.force.Cf_config import CfConfig
from cfdmod.use_cases.pressure.force.Cf_geom import get_representative_areas
from cfdmod.use_cases.pressure.geometry import (
    GeometryData,
    ProcessedEntity,
    get_excluded_entities,
    get_geometry_data,
    get_region_definition_dataframe,
    tabulate_geometry_data,
)
from cfdmod.use_cases.pressure.output import CommonOutput
from cfdmod.use_cases.pressure.zoning.body_config import BodyDefinition
from cfdmod.use_cases.pressure.zoning.processing import (
    calculate_statistics,
    combine_stats_data_with_mesh,
)
from cfdmod.utils import convert_dataframe_into_matrix


def process_Cf(
    mesh: LnasFormat,
    cfg: CfConfig,
    cp_path: pathlib.Path,
    bodies_definition: dict[str, BodyDefinition],
) -> dict[str, CommonOutput]:
    """Executes the force coefficient processing routine

    Args:
        mesh (LnasFormat): Input mesh
        cfg (CfConfig): Force coefficient configuration
        cp_path (pathlib.Path): Path for pressure coefficient time series
        bodies_definition (dict[str, BodyDefinition]): Dictionary of bodies definition

    Returns:
        dict[str, CommonOutput]: Compiled outputs for force coefficient use case keyed by direction
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

    Cf_data = process_timestep_groups(
        data_path=cp_path,
        geometry_df=geometry_df,
        geometry=geometry_to_use,
        processing_function=transform_Cf,
    )

    region_definition_df = get_region_definition_dataframe(geometry_dict)
    length_df = Cf_data[["region_idx", "Lx", "Ly", "Lz"]].drop_duplicates()
    Cf_data.drop(columns=["Lx", "Ly", "Lz"], inplace=True)

    region_definition_df = pd.merge(
        region_definition_df,
        length_df,
        on="region_idx",
        how="left",
    )

    included_sfc_list = [
        sfc for body_cfg in cfg.bodies for sfc in bodies_definition[body_cfg.name].surfaces
    ]
    excluded_sfc_list = [sfc for sfc in mesh.surfaces.keys() if sfc not in included_sfc_list]

    if len(excluded_sfc_list) != 0 and len(included_sfc_list) != 0:
        col = [s.stats for s in cfg.statistics]
        excluded_entities = [
            get_excluded_entities(excluded_sfc_list=excluded_sfc_list, mesh=mesh, data_columns=col)
        ]
    else:
        excluded_entities = []

    compild_cf_output = {}
    for direction_lbl in cfg.directions:
        Cf_dir_data = convert_dataframe_into_matrix(
            Cf_data[["region_idx", "time_normalized", f"Cf{direction_lbl}"]],
            row_data_label="time_normalized",
            column_data_label="region_idx",
            value_data_label=f"Cf{direction_lbl}",
        )

        Cf_stats = calculate_statistics(
            historical_data=Cf_dir_data, statistics_to_apply=cfg.statistics
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
                data_stats=Cf_stats,
            )
            polydata = create_polydata_for_cell_data(body_data_df, body_data.mesh)
            data_entity = ProcessedEntity(mesh=body_data.mesh, polydata=polydata)
            processed_entities.append(data_entity)

        compild_cf_output[direction_lbl] = CommonOutput(
            data_df=Cf_dir_data,
            stats_df=Cf_stats,
            processed_entities=processed_entities,
            excluded_entities=excluded_entities,
            region_indexing_df=geometry_df[["region_idx", "point_idx"]],
            region_definition_df=region_definition_df,
        )

    return compild_cf_output


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
    time_normalized = raw_cp["time_normalized"].copy()
    cols_points = [c for c in raw_cp.columns if c != "time_normalized"]
    id_points = np.array([int(c) for c in cols_points])

    points_selection = geometry_df.sort_values(by="point_idx")["point_idx"].to_numpy()
    face_area = geometry_df["area"].to_numpy()
    face_ns = geometry_df[["n_x", "n_y", "n_z"]].to_numpy().T

    mask_valid_points = np.isin(id_points, points_selection)
    id_points_selected = id_points[mask_valid_points]
    cp_matrix = raw_cp[cols_points].copy().to_numpy()[:, mask_valid_points]

    regions_list = geometry_df["region_idx"].unique()

    f_matrix_x = -cp_matrix * face_area * face_ns[0, :]
    f_matrix_y = -cp_matrix * face_area * face_ns[1, :]
    f_matrix_z = -cp_matrix * face_area * face_ns[2, :]

    list_of_cf_region = []
    for region in regions_list:
        points_of_region = geometry_df[geometry_df["region_idx"] == region]["point_idx"].to_numpy()
        mask_points_of_region = np.isin(id_points_selected, points_of_region)

        cf_region = pd.DataFrame(
            {
                "time_normalized": time_normalized,
                "fx": np.sum(f_matrix_x[:, mask_points_of_region], axis=1),
                "fy": np.sum(f_matrix_y[:, mask_points_of_region], axis=1),
                "fz": np.sum(f_matrix_z[:, mask_points_of_region], axis=1),
                "region_idx": region,
            }
        )
        list_of_cf_region.append(cf_region)

    cf_full = pd.concat(list_of_cf_region)
    del list_of_cf_region

    Cf_data = (
        cf_full.groupby(["region_idx", "time_normalized"])  # type: ignore
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
        (Lx, Ly, Lz), (Ax, Ay, Az) = get_representative_areas(
            input_mesh=geometry, point_idx=region_points_idx
        )

        representative_areas[region_idx[0]] = {}
        representative_areas[region_idx[0]]["ATx"] = Ax
        representative_areas[region_idx[0]]["ATy"] = Ay
        representative_areas[region_idx[0]]["ATz"] = Az
        representative_areas[region_idx[0]]["Lx"] = Lx
        representative_areas[region_idx[0]]["Ly"] = Ly
        representative_areas[region_idx[0]]["Lz"] = Lz

    rep_df = pd.DataFrame.from_dict(representative_areas, orient="index").reset_index()
    rep_df = rep_df.rename(columns={"index": "region_idx"})
    Cf_data = pd.merge(Cf_data, rep_df, on="region_idx")

    Cf_data["Cfx"] = Cf_data["Fx"] / Cf_data["ATx"]
    Cf_data["Cfy"] = Cf_data["Fy"] / Cf_data["ATy"]
    Cf_data["Cfz"] = Cf_data["Fz"] / Cf_data["ATz"]

    Cf_data.drop(columns=["Fx", "Fy", "Fz", "ATx", "ATy", "ATz"], inplace=True)

    return Cf_data
