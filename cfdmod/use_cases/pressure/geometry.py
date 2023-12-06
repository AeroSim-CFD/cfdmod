import numpy as np
import pandas as pd
from lnas import LnasFormat, LnasGeometry
from vtk import vtkPolyData

from cfdmod.api.vtk.write_vtk import create_polydata_for_cell_data
from cfdmod.use_cases.pressure.zoning.body_config import BodyConfig


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


def get_geometry_from_mesh(
    body_cfg: BodyConfig, mesh: LnasFormat
) -> tuple[LnasGeometry, np.ndarray]:
    """Filters the mesh from the list of surfaces that define the body in config

    Args:
        body_cfg (BodyConfig): Body configuration
        mesh (LnasFormat): LNAS mesh

    Raises:
        Exception: Surface specified is not defined in LNAS

    Returns:
        tuple[LnasGeometry, np.ndarray]: Tuple containing the body geometry and the filtered triangle indexes
    """
    if len(body_cfg.surfaces) == 0:
        # Include all surfaces
        geometry_idx = np.arange(0, len(mesh.geometry.triangles))
    else:
        # Filter mesh for all surfaces
        geometry_idx = np.array([], dtype=np.int32)
        for sfc in body_cfg.surfaces:
            if sfc not in mesh.surfaces.keys():
                raise Exception(
                    f"Surface {sfc} defined in body is not separated in the LNAS file."
                )
            geometry_idx = np.concatenate((geometry_idx, mesh.surfaces[sfc]))

    boolean_array = np.full(len(mesh.geometry.triangles), False, dtype=bool)
    boolean_array[geometry_idx] = True
    filtered_format = mesh.filter_triangles(boolean_array)

    return filtered_format.geometry, np.unique(geometry_idx)
