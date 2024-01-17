from dataclasses import dataclass

import numpy as np
import pandas as pd
from lnas import LnasFormat, LnasGeometry
from vtk import vtkPolyData

from cfdmod.api.geometry.transformation_config import TransformationConfig
from cfdmod.api.vtk.write_vtk import create_polydata_for_cell_data
from cfdmod.use_cases.pressure.shape.zoning_config import ZoningModel
from cfdmod.use_cases.pressure.zoning.processing import get_indexing_mask


@dataclass
class GeometryData:
    mesh: LnasGeometry
    zoning_to_use: ZoningModel
    triangles_idxs: np.ndarray


def get_excluded_surfaces(mesh: LnasFormat, sfc_list: list[str]) -> LnasGeometry:
    """Filters the surfaces that were excluded in processing

    Args:
        mesh (LnasFormat): LNAS body mesh
        sfc_list (list[str]): List of excluded surfaces

    Returns:
        LnasGeometry: Returns a LnasGeometry if any surface was excluded
    """
    excluded_ids = np.array([], dtype=np.uint32)
    for excluded_sfc in sfc_list:
        if not excluded_sfc in mesh.surfaces.keys():
            raise Exception("Surface is not defined in LNAS.")
        ids = mesh.surfaces[excluded_sfc].copy()
        excluded_ids = np.concatenate((excluded_ids, ids))

    if excluded_ids.size != 0:
        excluded_geom = LnasGeometry(
            vertices=mesh.geometry.vertices.copy(),
            triangles=mesh.geometry.triangles[excluded_ids].copy(),
        )
        return excluded_geom
    else:
        raise Exception("No geometry could be filtered from the list of surfaces.")


def create_NaN_polydata(mesh: LnasGeometry, column_labels: list[str]) -> vtkPolyData:
    """Creates vtkPolyData from a given mesh and populate column labels with NaN values

    Args:
        mesh (LnasGeometry): Input LNAS mesh
        column_labels (list[str]): Column labels to populate with NaN values

    Returns:
        vtkPolyData: Polydata with the input mesh and NaN values
    """
    mock_df = pd.DataFrame(columns=column_labels)
    mock_df["point_idx"] = np.arange(0, mesh.triangles.shape[0])
    # All other columns will be NaN except for point_idx
    polydata = create_polydata_for_cell_data(data=mock_df, mesh=mesh)

    return polydata


def filter_geometry_from_list(
    mesh: LnasFormat, sfc_list: list[str]
) -> tuple[LnasGeometry, np.ndarray]:
    """Filters the mesh from a list of surfaces

    Args:
        mesh (LnasFormat): LNAS mesh with every surface available
        sfc_list (list[str]): List of surfaces to be filtered

    Returns:
        tuple[LnasGeometry, np.ndarray]: Tuple with filtered LNAS mesh geometry
        and the filtered triangle indices
    """
    geom_mesh = LnasGeometry(
        vertices=mesh.geometry.vertices, triangles=np.empty((0, 3), dtype=np.uint32)
    )
    geom_triangles_idxs = np.array([], dtype=np.uint32)

    for sfc in sfc_list:
        m = mesh.geometry_from_surface(sfc)
        geom_mesh.triangles = np.vstack((geom_mesh.triangles, m.triangles))
        geom_triangles_idxs = np.hstack((geom_triangles_idxs, mesh.surfaces[sfc].copy()))

    geom_mesh._full_update()

    return geom_mesh, geom_triangles_idxs


def get_region_indexing(
    geom_data: GeometryData,
    transformation: TransformationConfig,
) -> np.ndarray:
    """Index each triangle from the geometry after applying transformation

    Args:
        geom_data (GeometryData): Geometry data
        transformation (TransformationConfig): Transformation configuration

    Returns:
        np.ndarray: Triangle indexing. Each triangle of the geometry has a corresponding region index
    """
    df_regions = geom_data.zoning_to_use.get_regions_df()

    transformed_geometry = geom_data.mesh.copy()
    transformed_geometry.apply_transformation(transformation.get_geometry_transformation())

    triangles_region_idx = get_indexing_mask(mesh=transformed_geometry, df_regions=df_regions)

    return triangles_region_idx


def combine_geometries(geometries_list: list[LnasGeometry]) -> LnasGeometry:
    """Combine a list of LnasGeometry into a single LnasGeometry

    Args:
        geometries_list (list[LnasGeometry]): List of LnasGeometry to be combined

    Returns:
        LnasGeometry: Result of the combination of a list of LnasGeometry
    """
    result_geometry = LnasGeometry(
        vertices=np.empty((0, 3), dtype=np.uint32), triangles=np.empty((0, 3), dtype=np.uint32)
    )

    for geometry in geometries_list:
        geometry.triangles += len(result_geometry.vertices)
        result_geometry.vertices = np.vstack((result_geometry.vertices, geometry.vertices))
        result_geometry.triangles = np.vstack((result_geometry.triangles, geometry.triangles))

    result_geometry._full_update()

    return result_geometry
