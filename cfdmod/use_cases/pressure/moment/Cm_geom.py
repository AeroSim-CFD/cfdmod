import numpy as np
import pandas as pd

from cfdmod.api.geometry.transformation_config import TransformationConfig
from cfdmod.use_cases.pressure.geometry import GeometryData


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
