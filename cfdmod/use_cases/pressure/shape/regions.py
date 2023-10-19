import numpy as np
import pandas as pd
from nassu.lnas import LagrangianGeometry


def get_region_index_mask(mesh: LagrangianGeometry, df_regions: pd.DataFrame) -> np.ndarray:
    """Index the region of each triangle in the mesh

    Args:
        mesh (LagrangianGeometry): Mesh with triangles to label
        df_regions (pd.DataFrame): Dataframe describing the regions intervals (x_min, x_max, y_min, y_max, z_min, z_max, region_index)

    Returns:
        np.ndarray: Triangles region indexing array
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
