from typing import Literal, Optional

import numpy as np
import pandas as pd
from lnas import LnasGeometry

from cfdmod.use_cases.pressure.extreme_values import (
    ExtremeValuesParameters,
    calculate_extreme_values,
)
from cfdmod.use_cases.pressure.statistics import Statistics

ForceVariables = Literal["Cfx", "Cfy", "Cfz"]
MomentVariables = Literal["Cmx", "Cmy", "Cmz"]
ShapeVariables = Literal["Ce"]


def get_indexing_mask(mesh: LnasGeometry, df_regions: pd.DataFrame) -> np.ndarray:
    """Index each triangle in the mesh in the respective region

    Args:
        mesh (LnasGeometry): Mesh with triangles to index
        df_regions (pd.DataFrame): Dataframe describing the zoning intervals (x_min, x_max, y_min, y_max, z_min, z_max, region_idx)

    Returns:
        np.ndarray: Triangles zoning indexing array
    """
    triangles = mesh.triangle_vertices
    centroids = np.mean(triangles, axis=1)

    triangles_region = np.full((triangles.shape[0],), -1, dtype=np.int32)

    for _, region in df_regions.iterrows():
        ll = np.array([region["x_min"], region["y_min"], region["z_min"]])  # lower-left
        ur = np.array([region["x_max"], region["y_max"], region["z_max"]])  # upper-right

        in_idx = np.all(
            np.logical_and(
                centroids >= ll,
                centroids < ur,
            ),
            axis=1,
        )
        triangles_region[in_idx] = region["region_idx"]

    return triangles_region


def calculate_statistics(
    historical_data: pd.DataFrame,
    statistics_to_apply: list[Statistics],
    variables: list[ShapeVariables] | list[ForceVariables] | list[MomentVariables],
    extreme_params: Optional[ExtremeValuesParameters] = None,
) -> pd.DataFrame:
    """Calculates statistics for force coefficient of a body data

    Args:
        historical_data (pd.DataFrame): Dataframe of the data coefficients historical series
        statistics_to_apply (list[Statistics]): List of statistical functions to apply
        variables (list[str]): List of variables to apply statistical analysis
        extreme_params (Optional[ExtremeValuesParameters]): Parameters for extreme values analysis. Defaults to None.

    Returns:
        pd.DataFrame: Statistics for the given coefficient
    """
    group_by_point = historical_data.groupby("region_idx")

    statistics_data = pd.DataFrame({"region_idx": historical_data["region_idx"].unique()})

    for var in variables:
        if "mean" in statistics_to_apply:
            average = group_by_point[var].apply(lambda x: x.mean()).reset_index(name="mean")
            statistics_data[f"{var}_mean"] = average["mean"]
        if "min" in statistics_to_apply:
            minimum = group_by_point[var].apply(lambda x: x.min()).reset_index(name="min")
            statistics_data[f"{var}_min"] = minimum["min"]
        if "max" in statistics_to_apply:
            maximum = group_by_point[var].apply(lambda x: x.max()).reset_index(name="max")
            statistics_data[f"{var}_max"] = maximum["max"]
        if "std" in statistics_to_apply:
            std = group_by_point[var].apply(lambda x: x.std()).reset_index(name="std")
            statistics_data[f"{var}_std"] = std["std"]

        # Calculate skewness and kurtosis using apply
        if "skewness" in statistics_to_apply:
            skewness = group_by_point[var].apply(lambda x: x.skew()).reset_index(name="skewness")
            statistics_data[f"{var}_skewness"] = skewness["skewness"]
        if "kurtosis" in statistics_to_apply:
            kurtosis = group_by_point[var].apply(lambda x: x.kurt()).reset_index(name="kurtosis")
            statistics_data[f"{var}_kurtosis"] = kurtosis["kurtosis"]

        # Extreme values analysis
        if any([v in statistics_to_apply for v in ["xtr_min", "xtr_max"]]):
            if extreme_params is None:
                raise ValueError("Missing extreme values parameters!")
            timestep = pd.unique(historical_data.time_step).to_numpy(dtype=np.float32)
            xtr_stats = (
                group_by_point[var]
                .apply(
                    lambda x: calculate_extreme_values(
                        params=extreme_params, timestep_arr=timestep, hist_series=x
                    )
                )
                .reset_index(name="xtr_val")
            )
            statistics_data[[f"{var}_xtr_min", f"{var}_xtr_max"]] = xtr_stats["xtr_val"].apply(
                lambda x: pd.Series(x)
            )
            if "xtr_min" not in statistics_to_apply:
                statistics_data = statistics_data.drop(f"{var}_xtr_min", axis=1)
            if "xtr_max" not in statistics_to_apply:
                statistics_data = statistics_data.drop(f"{var}_xtr_max", axis=1)

    return statistics_data


def combine_stats_data_with_mesh(
    mesh: LnasGeometry,
    region_idx_array: np.ndarray,
    data_stats: pd.DataFrame,
) -> pd.DataFrame:
    """Combine compiled statistical data with surface meshing by indexing regions

    Args:
        mesh (LnasGeometry): LNAS mesh to be combined
        region_idx_array (np.ndarray): Triangles indexing by region
        data_stats (pd.DataFrame): Compiled statistics data

    Returns:
        pd.DataFrame: Dataframe with region statistics indexed by mesh triangles
    """
    combined_df = pd.DataFrame()
    combined_df["point_idx"] = np.arange(len(mesh.triangle_vertices))
    combined_df["region_idx"] = region_idx_array
    combined_df = pd.merge(combined_df, data_stats, on="region_idx", how="left")
    combined_df.drop(columns=["region_idx"], inplace=True)

    return combined_df
