import numpy as np
import pandas as pd
from nassu.lnas import LagrangianGeometry


def filter_surface_data(cp_data):
    ...


def get_sub_body_index_mask(mesh: LagrangianGeometry, df_regions: pd.DataFrame) -> np.ndarray:
    """Index the sub body of each triangle in the mesh

    Args:
        mesh (LagrangianGeometry): Mesh with triangles to index
        df_regions (pd.DataFrame): Dataframe describing the sub body intervals (x_min, x_max, y_min, y_max, z_min, z_max, sub_body_idx)

    Returns:
        np.ndarray: Triangles sub body indexing array
    """
    triangles = mesh.triangle_vertices
    centroids = np.mean(triangles, axis=1)

    triangles_region = np.full((triangles.shape[0],), -1, dtype=np.int32)

    for index, region in df_regions.iterrows():
        ll = np.array([region["x_min"], region["y_min"], region["z_min"]])  # lower-left
        ur = np.array([region["x_max"], region["y_max"], region["z_max"]])  # upper-right

        in_idx = np.all(
            np.logical_and(
                centroids >= ll,
                centroids < ur,
            ),
            axis=1,
        )
        triangles_region[in_idx] = region["region_index"]

    return triangles_region
