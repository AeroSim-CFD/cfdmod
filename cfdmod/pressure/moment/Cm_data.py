import pathlib

import numpy as np
import pandas as pd
from lnas import LnasFormat, LnasGeometry

from cfdmod.io.vtk.write_vtk import create_polydata_for_cell_data
from cfdmod.pressure.chunking import process_timestep_groups
from cfdmod.pressure.geometry import (
    GeometryData,
    ProcessedEntity,
    get_excluded_entities,
    get_geometry_data,
    get_region_definition_dataframe,
    tabulate_geometry_data,
)
from cfdmod.pressure.moment.Cm_config import CmConfig
from cfdmod.io.geometry.transformation_config import TransformationConfig
from cfdmod.pressure.output import CommonOutput
from cfdmod.pressure.zoning.body_config import BodyDefinition
from cfdmod.pressure.zoning.processing import (
    calculate_statistics,
    combine_stats_data_with_mesh,
)
from cfdmod.utils import convert_dataframe_into_matrix


def add_lever_arm_to_geometry_df(
    geom_data: GeometryData,
    transformation: TransformationConfig,
    lever_origin: tuple[float, float, float],
    geometry_df: pd.DataFrame,
) -> pd.DataFrame:
    """Adds a value for the lever arm for each point for moment calculations

    Args:
        geom_data (GeometryData): Geometry data object
        transformation (TransformationConfig): Transformation config to apply to the geometry
        lever_origin (tuple[float, float, float]): Origin of the lever after the geometry is transformed
        geometry_df (pd.DataFrame): Dataframe with geometric properties

    Returns:
        pd.DataFrame: Merged geometry_df with lever arm lengths
    """
    transformed_body = geom_data.mesh.copy()
    transformed_body.apply_transformation(transformation.get_geometry_transformation())
    centroids = np.mean(transformed_body.triangle_vertices, axis=1)

    position_df = _get_lever_relative_position_df(
        centroids=centroids, lever_origin=lever_origin, geometry_idx=geom_data.triangles_idxs
    )
    result = pd.merge(geometry_df, position_df, on="point_idx", how="left")

    return result


def _get_lever_relative_position_df(
    centroids: np.ndarray, lever_origin: tuple[float, float, float], geometry_idx: np.ndarray
) -> pd.DataFrame:
    """Creates a Dataframe with the relative position for each triangle

    Args:
        centroids (np.ndarray): Array of triangle centroids
        lever_origin (tuple[float, float, float]): Coordinate of the lever origin
        geometry_idx (np.ndarray): Indexes of the triangles of the body

    Returns:
        pd.DataFrame: Relative position dataframe
    """
    position_df = pd.DataFrame()
    position_df["rx"] = centroids[:, 0] - lever_origin[0]
    position_df["ry"] = centroids[:, 1] - lever_origin[1]
    position_df["rz"] = centroids[:, 2] - lever_origin[2]
    position_df["point_idx"] = geometry_idx

    return position_df


def get_representative_volume(
    input_mesh: LnasGeometry, point_idx: np.ndarray
) -> tuple[tuple[float, float, float], float]:
    """Calculates the representative volume from the bounding box of a given mesh

    Args:
        input_mesh (LnasGeometry): Input LNAS mesh
        point_idx (np.ndarray): Array of triangle indices of each sub region

    Returns:
        tuple[tuple[float, float, float], float]: Tuple containing:
            Lengths tuple (Lx, Ly, Lz) and representative volume value
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

    V_rep = Lx * Ly * Lz

    return (Lx, Ly, Lz), V_rep


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

    def wrapper_transform_Cm(
        raw_cp: pd.DataFrame, geometry_df: pd.DataFrame, geometry: LnasGeometry
    ):
        return transform_Cm(raw_cp, geometry_df, geometry, nominal_volume=cfg.nominal_volume)

    Cm_data = process_timestep_groups(
        data_path=cp_path,
        geometry_df=geometry_df,
        geometry=geometry_to_use,
        processing_function=wrapper_transform_Cm,
    )
    region_definition_df = get_region_definition_dataframe(geometry_dict)
    length_df = Cm_data[["region_idx"]].drop_duplicates()
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
            region_definition_df=region_definition_df,
        )

    return compild_cm_output


def transform_Cm(
    raw_cp: pd.DataFrame,
    geometry_df: pd.DataFrame,
    geometry: LnasGeometry,
    *,
    nominal_volume: float,
) -> pd.DataFrame:
    """Transforms pressure coefficient into moment coefficient

    Args:
        raw_cp (pd.DataFrame): Body pressure coefficient data
        geometry_df (pd.DataFrame): Dataframe with geometric properties and triangle indexing
        geometry (LnasGeometry): Mesh geometry for bounding box definition
        nominal_volume (float): Nominal volume to use for moment coeficient

    Returns:
        pd.DataFrame: Moment coefficient dataframe
    """
    time_normalized = raw_cp["time_normalized"].copy()
    cols_points = [c for c in raw_cp.columns if c != "time_normalized"]
    id_points = np.array([int(c) for c in cols_points])

    points_selection = geometry_df.sort_values(by="point_idx")["point_idx"].to_numpy()
    face_area = geometry_df["area"].to_numpy()
    face_ns = geometry_df[["n_x", "n_y", "n_z"]].to_numpy().T
    face_pos = geometry_df[["rx", "ry", "rz"]].to_numpy().T

    mask_valid_points = np.isin(id_points, points_selection)
    id_points_selected = id_points[mask_valid_points]
    cp_matrix = raw_cp[cols_points].copy().to_numpy()[:, mask_valid_points]

    regions_list = geometry_df["region_idx"].unique()

    f_matrix_x = -cp_matrix * face_area * face_ns[0, :]
    f_matrix_y = -cp_matrix * face_area * face_ns[1, :]
    f_matrix_z = -cp_matrix * face_area * face_ns[2, :]

    m_matrix_x = face_pos[1, :] * f_matrix_z - face_pos[2, :] * f_matrix_y
    m_matrix_y = face_pos[2, :] * f_matrix_x - face_pos[0, :] * f_matrix_z
    m_matrix_z = face_pos[0, :] * f_matrix_y - face_pos[1, :] * f_matrix_x

    list_of_cm_region = []
    for region in regions_list:
        points_of_region = geometry_df[geometry_df["region_idx"] == region]["point_idx"].to_numpy()
        mask_points_of_region = np.isin(id_points_selected, points_of_region)

        cm_region = pd.DataFrame(
            {
                "time_normalized": time_normalized,
                "mx": np.sum(m_matrix_x[:, mask_points_of_region], axis=1),
                "my": np.sum(m_matrix_y[:, mask_points_of_region], axis=1),
                "mz": np.sum(m_matrix_z[:, mask_points_of_region], axis=1),
                "region_idx": region,
            }
        )
        list_of_cm_region.append(cm_region)

    cm_full = pd.concat(list_of_cm_region)
    del list_of_cm_region

    Cm_data = (
        cm_full.groupby(["region_idx", "time_normalized"])  # type: ignore
        .agg(
            Mx=pd.NamedAgg(column="mx", aggfunc="sum"),
            My=pd.NamedAgg(column="my", aggfunc="sum"),
            Mz=pd.NamedAgg(column="mz", aggfunc="sum"),
        )
        .reset_index()
    )

    if nominal_volume > 0:
        Cm_data["Cmx"] = Cm_data["Mx"] / nominal_volume
        Cm_data["Cmy"] = Cm_data["My"] / nominal_volume
        Cm_data["Cmz"] = Cm_data["Mz"] / nominal_volume

        Cm_data.drop(columns=["Mx", "My", "Mz"], inplace=True)
    else:
        region_group_by = geometry_df.groupby(["region_idx"])
        representative_volume = {}

        for region_idx, region_points in region_group_by:
            region_points_idx = region_points.point_idx.to_numpy()
            (Lx, Ly, Lz), V_rep = get_representative_volume(
                input_mesh=geometry, point_idx=region_points_idx
            )

            representative_volume[region_idx[0]] = {}
            representative_volume[region_idx[0]]["V_rep"] = V_rep
            representative_volume[region_idx[0]]["Lx"] = Lx
            representative_volume[region_idx[0]]["Ly"] = Ly
            representative_volume[region_idx[0]]["Lz"] = Lz

        rep_df = pd.DataFrame.from_dict(representative_volume, orient="index").reset_index()
        rep_df = rep_df.rename(columns={"index": "region_idx"})

        Cm_data = pd.merge(Cm_data, rep_df, on="region_idx")

        Cm_data["Cmx"] = Cm_data["Mx"] / Cm_data["V_rep"]
        Cm_data["Cmy"] = Cm_data["My"] / Cm_data["V_rep"]
        Cm_data["Cmz"] = Cm_data["Mz"] / Cm_data["V_rep"]

        Cm_data.drop(columns=["Mx", "My", "Mz", "V_rep"], inplace=True)

    return Cm_data
