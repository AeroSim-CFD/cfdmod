"""Core computation functions for the pressure module.

Merges logic from:
  cp_data.py, force/Cf_data.py, moment/Cm_data.py, shape/Ce_data.py,
  zoning/processing.py, extreme_values.py
"""

from __future__ import annotations

__all__ = [
    # Cp
    "add_cp2xdmf",
    "process_xdmf_to_cp",
    "process_timeseries",
    # Cf
    "get_representative_areas",
    "transform_Cf",
    "process_Cf",
    # Cm
    "add_lever_arm_to_geometry_df",
    "get_representative_volume",
    "transform_Cm",
    "process_Cm",
    # Ce
    "transform_Ce",
    "get_surface_dict",
    "process_surfaces",
    "process_Ce",
    # Zoning helpers (re-exported from geometry)
    "get_indexing_mask",
    "combine_stats_data_with_mesh",
    # Statistics helpers
    "extreme_values_analysis",
    "calculate_extreme_values",
    "calculate_mean_equivalent",
    "calculate_statistics",
    # Extreme values
    "fit_gumbel_model",
    "gumbel_extreme_values",
    "moving_average_extreme_values",
    "peak_extreme_values",
]

import math
import pathlib
import warnings
from dataclasses import dataclass

import h5py
import numpy as np
import pandas as pd
from lnas import LnasFormat, LnasGeometry

from cfdmod.io.xdmf import (
    get_pressure_keys,
    filter_keys_by_range,
    read_timeseries_meta,
    write_timeseries_geometry,
    write_timeseries_meta,
    write_timeseries_step,
    write_temporal_xdmf,
)
from cfdmod.logger import logger
from cfdmod.pressure.geometry import (
    GeometryData,
    ProcessedEntity,
    combine_stats_data_with_mesh,
    generate_regions_mesh,
    get_ce_geometry_data,
    get_geometry_data,
    get_indexing_mask,
    get_region_definition_dataframe,
    tabulate_geometry_data,
)
from cfdmod.pressure.parameters import (
    BasicStatisticModel,
    CeConfig,
    CfConfig,
    CmConfig,
    CpConfig,
    ExtremeGumbelParamsModel,
    ExtremeMovingAverageParamsModel,
    ExtremePeakParamsModel,
    ParameterizedStatisticModel,
    StatisticsParamsModel,
    BodyDefinition,
)
from cfdmod.utils import convert_dataframe_into_matrix, create_folders_for_file


# ---------------------------------------------------------------------------
# CommonOutput
# ---------------------------------------------------------------------------


@dataclass
class CommonOutput:
    """Compiled output for a coefficient computation."""

    data_df: pd.DataFrame
    stats_df: pd.DataFrame
    region_indexing_df: pd.DataFrame
    region_definition_df: pd.DataFrame


@dataclass
class CeOutput(CommonOutput):
    """Shape coefficient output, includes region meshes for export."""

    processed_entities: list[ProcessedEntity]
    excluded_entities: list[ProcessedEntity]

    def export_mesh(self, cfg_label: str, path_manager):
        regions_mesh = self.processed_entities[0].mesh.copy()
        regions_mesh.join([sfc.mesh.copy() for sfc in self.processed_entities[1:]])
        mesh_path = path_manager.get_surface_path(cfg_lbl=cfg_label, sfc_lbl="body")
        create_folders_for_file(mesh_path)
        regions_mesh.export_stl(mesh_path)

        if len(self.excluded_entities) != 0:
            excluded_path = path_manager.get_surface_path(
                cfg_lbl=cfg_label, sfc_lbl="excluded_surfaces"
            )
            self.excluded_entities[0].mesh.export_stl(excluded_path)


# ---------------------------------------------------------------------------
# Extreme values (formerly extreme_values.py)
# ---------------------------------------------------------------------------


def fit_gumbel_model(
    data: np.ndarray, params: ExtremeGumbelParamsModel, sample_duration: float
) -> float:
    """Fit the Gumbel model to predict extreme events."""
    N = len(data)
    y = [-math.log(-math.log(i / (N + 1))) for i in range(1, N + 1)]
    A = np.vstack([y, np.ones(len(y))]).T
    a_inv, U_T0 = np.linalg.lstsq(A, data, rcond=None)[0]
    U_T1 = U_T0 + a_inv * math.log(params.event_duration / (sample_duration / N))
    extreme_val = a_inv * params.yR + U_T1
    return extreme_val


def gumbel_extreme_values(
    params: ExtremeGumbelParamsModel,
    timestep_arr: np.ndarray,
    hist_series: np.ndarray,
) -> tuple[float, float]:
    """Apply Gumbel extreme values analysis to a coefficient historic series."""
    CST_full_scale = params.full_scale_characteristic_length / params.full_scale_U_H
    time = (timestep_arr - timestep_arr[0]) * CST_full_scale
    T0 = time[-1]
    window_size = max(int(params.peak_duration / (time[1] - time[0])), 1)
    smooth_parent_cp = np.convolve(
        hist_series, np.ones(window_size) / window_size, mode="valid"
    )
    sub_arrays = np.array_split(smooth_parent_cp, params.n_subdivisions)
    cp_max = np.sort(np.array([np.max(sub) for sub in sub_arrays]))
    cp_min = np.sort(np.array([np.min(sub) for sub in sub_arrays]))[::-1]
    max_val = fit_gumbel_model(cp_max, params=params, sample_duration=T0)
    min_val = fit_gumbel_model(cp_min, params=params, sample_duration=T0)
    min_val = 0 if np.isnan(min_val) else min_val
    max_val = 0 if np.isnan(max_val) else max_val
    return min_val, max_val


def moving_average_extreme_values(
    params: ExtremeMovingAverageParamsModel, hist_series: np.ndarray
) -> tuple[float, float]:
    """Apply moving average extreme values analysis."""
    CST_full_scale = params.full_scale_characteristic_length / params.full_scale_U_H
    window_size = max(1, round(params.window_size_interval / CST_full_scale))
    kernel = np.ones(window_size) / window_size
    smoothed = np.convolve(hist_series, kernel, mode="valid")
    return smoothed.min(), smoothed.max()


def peak_extreme_values(
    params: ExtremePeakParamsModel, hist_series: np.ndarray
) -> tuple[float, float]:
    """Apply peak factor extreme values analysis."""
    avg = hist_series.mean()
    std = hist_series.std()
    return avg - params.peak_factor * std, avg + params.peak_factor * std


# ---------------------------------------------------------------------------
# Statistics helpers (formerly zoning/processing.py)
# ---------------------------------------------------------------------------


def extreme_values_analysis(
    params: StatisticsParamsModel,
    data_df: pd.DataFrame,
    timestep_arr: np.ndarray,
) -> pd.DataFrame:
    """Perform extreme values analysis to a dataframe."""
    if params.method_type == "Absolute":
        return data_df.apply(lambda x: (x.min(), x.max()))
    elif params.method_type == "Gumbel":
        return data_df.apply(
            lambda x: gumbel_extreme_values(
                params=params, timestep_arr=timestep_arr, hist_series=x
            )
        )
    elif params.method_type == "Peak":
        return data_df.apply(
            lambda x: peak_extreme_values(params=params, hist_series=x)
        )
    elif params.method_type == "Moving Average":
        return data_df.apply(
            lambda x: moving_average_extreme_values(params=params, hist_series=x)
        )
    raise ValueError(f"Unknown method_type: {params.method_type}")


def calculate_extreme_values(
    extreme_statistics: list[ParameterizedStatisticModel],
    timestep_arr: np.ndarray,
    data_df: pd.DataFrame,
) -> dict[str, pd.DataFrame]:
    """Calculate extreme values from historical data."""
    stats_df_dict = {}
    stats = [s for s in extreme_statistics if s.stats in ["min", "max"]]
    if (
        len(set([s.stats for s in stats])) == len(stats) == 2
        and len(set([s.params.method_type for s in stats])) == 1
    ):
        extremes_df = extreme_values_analysis(
            params=stats[0].params,
            data_df=data_df,
            timestep_arr=timestep_arr,
        )
        stats_df_dict["min"] = extremes_df.iloc[0]
        stats_df_dict["max"] = extremes_df.iloc[1]
    else:
        for stat in stats:
            extremes_df = extreme_values_analysis(
                params=stat.params,
                data_df=data_df,
                timestep_arr=timestep_arr,
            )
            target_index = 0 if stat.stats == "min" else 1
            stats_df_dict[stat.stats] = extremes_df.iloc[target_index]
    return stats_df_dict


def calculate_mean_equivalent(
    statistics_to_apply: list[BasicStatisticModel | ParameterizedStatisticModel],
    stats_df_dict: dict[str, pd.Series],
) -> np.ndarray:
    """Calculate mean equivalent values."""
    comparison_df = pd.DataFrame()
    mean_eq_stat = [s for s in statistics_to_apply if s.stats == "mean_eq"][0]
    scale_factor = mean_eq_stat.params.scale_factor
    for stat_lbl in ["min", "max", "mean"]:
        comparison_df[stat_lbl] = stats_df_dict[stat_lbl].copy()
        comparison_df[stat_lbl] *= 1 if stat_lbl == "mean" else scale_factor
    max_abs_col_index = np.abs(comparison_df.values).argmax(axis=1)
    return comparison_df.values[np.arange(len(comparison_df)), max_abs_col_index]


def calculate_statistics(
    historical_data: pd.DataFrame,
    statistics_to_apply: list[BasicStatisticModel | ParameterizedStatisticModel],
) -> pd.DataFrame:
    """Calculate statistics for coefficient historical series.

    Args:
        historical_data (pd.DataFrame): Matrix-form DataFrame with time_normalized column
        statistics_to_apply: List of statistics to compute

    Returns:
        pd.DataFrame: Statistics indexed by region/point (rows) and stat name (columns)
    """
    stats_df_dict: dict[str, pd.Series] = {}
    statistics_list = [s.stats for s in statistics_to_apply]
    data_df = historical_data.drop(columns=["time_normalized"])

    if "mean" in statistics_list:
        stats_df_dict["mean"] = data_df.mean()
    if "rms" in statistics_list:
        stats_df_dict["rms"] = data_df.std()
    if "skewness" in statistics_list:
        stats_df_dict["skewness"] = data_df.skew()
    if "kurtosis" in statistics_list:
        stats_df_dict["kurtosis"] = data_df.kurt()
    if "min" in statistics_list or "max" in statistics_list:
        stats = [s for s in statistics_to_apply if s.stats in ["min", "max"]]
        stats_df_dict = stats_df_dict | calculate_extreme_values(
            extreme_statistics=stats,
            timestep_arr=historical_data["time_normalized"].to_numpy(),
            data_df=data_df,
        )
    if "mean_eq" in statistics_list:
        stats_df_dict["mean_eq"] = calculate_mean_equivalent(
            statistics_to_apply=statistics_to_apply, stats_df_dict=stats_df_dict
        )

    return pd.DataFrame(stats_df_dict)


# ---------------------------------------------------------------------------
# Cp functions
# ---------------------------------------------------------------------------


def add_cp2xdmf(
    *,
    body_h5: pathlib.Path,
    atm_probe_h5: pathlib.Path | None,
    reference_vel: float,
    fluid_density: float,
) -> None:
    """Add pressure coefficient (Cp) to H5 compatible with XDMF format.

    Args:
        body_h5 (pathlib.Path): H5 file for body pressure timeseries
        atm_probe_h5 (pathlib.Path | None): H5 file for atmospheric pressure probe.
            If None, constant reference pressure of 0 is used.
        reference_vel (float): Reference velocity
        fluid_density (float): Fluid density
    """
    with h5py.File(body_h5, mode="a") as f_body:
        grp_abs = f_body["pressure"]
        grp_cp = f_body.require_group("cp")
        keys = list(grp_abs.keys())

        if atm_probe_h5 is None:
            for k in keys:
                cp = grp_abs[k][:] / (0.5 * fluid_density * reference_vel**2)
                if k in grp_cp:
                    del grp_cp[k]
                grp_cp[k] = cp
            return

        with h5py.File(atm_probe_h5) as f_atm:
            grp_atm = f_atm["pressure"]
            for k in keys:
                p_body = grp_abs[k][:]
                p_ref = grp_atm[k][0]
                cp = (p_body - p_ref) / (0.5 * fluid_density * reference_vel**2)
                if k in grp_cp:
                    del grp_cp[k]
                grp_cp[k] = cp


def process_xdmf_to_cp(
    body_h5: pathlib.Path,
    probe_h5: pathlib.Path | None,
    output_path: pathlib.Path,
    cp_config: CpConfig,
) -> None:
    """Read body+probe XDMF H5 per-timestep, compute Cp, write to output H5.

    Handles macroscopic_type rho (cs^2=1/3) and pressure.
    Writes /cp/{t_key}, /meta/..., /Triangles, /Geometry to output_path.
    Also writes a temporal XDMF alongside the H5.

    Args:
        body_h5 (pathlib.Path): Body pressure XDMF H5 (pressure/t{T} per timestep)
        probe_h5 (pathlib.Path | None): Atmospheric probe H5 (pressure/t{T} with shape (1,))
        output_path (pathlib.Path): Output H5 file path
        cp_config (CpConfig): Pressure coefficient configuration
    """
    if output_path.exists():
        warnings.warn(
            f"Output path {output_path} exists, deleting it.", RuntimeWarning
        )
        output_path.unlink()

    keys = get_pressure_keys(body_h5, "pressure")
    if cp_config.timestep_range:
        keys = filter_keys_by_range(keys, cp_config.timestep_range)

    dynamic_pressure = 0.5 * cp_config.fluid_density * cp_config.simul_U_H**2
    multiplier = 1.0 / 3.0 if cp_config.macroscopic_type == "rho" else 1.0
    time_scale = (
        cp_config.simul_characteristic_length
        / cp_config.simul_U_H
        * cp_config.time_scale_multiplier
    )

    with h5py.File(body_h5, "r") as f:
        triangles = f["Triangles"][:]
        vertices = f["Geometry"][:]
    write_timeseries_geometry(output_path, triangles, vertices)

    time_steps_arr: list[float] = []
    time_normalized_arr: list[float] = []

    with h5py.File(body_h5, "r") as f_body:
        probe_file = h5py.File(probe_h5, "r") if probe_h5 is not None else None
        try:
            for t_val, t_key in keys:
                p_body = f_body["pressure"][t_key][:].astype(np.float64)
                if probe_file is not None:
                    p_ref = float(probe_file["pressure"][t_key][0])
                else:
                    p_ref = 0.0
                cp = (p_body - p_ref) * (multiplier / dynamic_pressure)
                write_timeseries_step(output_path, "cp", t_key, cp)
                time_steps_arr.append(t_val)
                time_normalized_arr.append(t_val / time_scale)
        finally:
            if probe_file is not None:
                probe_file.close()

    write_timeseries_meta(
        output_path,
        np.array(time_steps_arr),
        np.array(time_normalized_arr),
    )

    xdmf_path = output_path.with_suffix(".xdmf")
    write_temporal_xdmf(output_path, xdmf_path, "cp")
    logger.info(f"Cp timeseries written to {output_path}")


def process_timeseries(
    cp_h5: pathlib.Path,
    geometry_df: pd.DataFrame,
    geometry: LnasGeometry,
    processing_function,
    timestep_range: tuple[float, float] | None = None,
    batch_size: int = 200,
) -> pd.DataFrame:
    """Read Cp from H5 per-batch, apply processing_function, return result DataFrame.

    Args:
        cp_h5 (pathlib.Path): Cp timeseries H5 (cp/t{T} per timestep)
        geometry_df (pd.DataFrame): Geometric properties dataframe
        geometry (LnasGeometry): Full mesh geometry
        processing_function: Function(raw_cp, geometry_df, geometry) -> pd.DataFrame
        timestep_range (tuple | None): Optional (t_min, t_max) filter
        batch_size (int): Number of timesteps to load per batch

    Returns:
        pd.DataFrame: Concatenated processing results
    """
    keys = get_pressure_keys(cp_h5, "cp")
    if timestep_range is not None:
        keys = filter_keys_by_range(keys, timestep_range)

    meta = read_timeseries_meta(cp_h5)
    t_norm_map = {t: tn for t, tn in zip(meta["time_steps"], meta["time_normalized"])}

    processed_samples: list[pd.DataFrame] = []

    with h5py.File(cp_h5, "r") as f_cp:
        grp_cp = f_cp["cp"]
        for batch_start in range(0, len(keys), batch_size):
            batch_keys = keys[batch_start : batch_start + batch_size]
            batch_data = np.stack(
                [grp_cp[t_key][:].astype(np.float64) for _, t_key in batch_keys]
            )
            n_tri = batch_data.shape[1]
            cols = [str(i) for i in range(n_tri)]
            batch_df = pd.DataFrame(batch_data, columns=cols)
            batch_df["time_normalized"] = [t_norm_map[t_val] for t_val, _ in batch_keys]

            result = processing_function(batch_df, geometry_df, geometry)
            processed_samples.append(result)

    merged = pd.concat(processed_samples)
    merged.rename(columns={col: str(col) for col in merged.columns}, inplace=True)

    sort_cols = [col for col in ["time_normalized", "region_idx"] if col in merged.columns]
    if "time_normalized" not in merged.columns:
        raise KeyError("Missing time_normalized column in processed data")
    merged.sort_values(by=sort_cols, inplace=True)

    return merged


# ---------------------------------------------------------------------------
# Force coefficient (Cf)
# ---------------------------------------------------------------------------


def get_representative_areas(
    input_mesh: LnasGeometry, point_idx: np.ndarray
) -> tuple[tuple[float, float, float], tuple[float, float, float]]:
    """Calculate representative areas from the bounding box of a given mesh.

    Args:
        input_mesh (LnasGeometry): Input LNAS mesh
        point_idx (np.ndarray): Triangle indices of each sub region

    Returns:
        tuple: ((Lx, Ly, Lz), (Ax, Ay, Az))
    """
    geom_verts = input_mesh.triangle_vertices[point_idx].reshape(-1, 3)
    x_min, x_max = geom_verts[:, 0].min(), geom_verts[:, 0].max()
    y_min, y_max = geom_verts[:, 1].min(), geom_verts[:, 1].max()
    z_min, z_max = geom_verts[:, 2].min(), geom_verts[:, 2].max()

    Lx = max(x_max - x_min, 1.0)
    Ly = max(y_max - y_min, 1.0)
    Lz = max(z_max - z_min, 1.0)

    return (Lx, Ly, Lz), (Ly * Lz, Lx * Lz, Lx * Ly)


def transform_Cf(
    raw_cp: pd.DataFrame,
    geometry_df: pd.DataFrame,
    geometry: LnasGeometry,
    *,
    nominal_area: float,
) -> pd.DataFrame:
    """Transform pressure coefficient into force coefficient.

    Args:
        raw_cp (pd.DataFrame): Body Cp data (matrix form)
        geometry_df (pd.DataFrame): Geometric properties and triangle indexing
        geometry (LnasGeometry): Mesh geometry for bounding box definition
        nominal_area (float): Nominal area for Cf calculation

    Returns:
        pd.DataFrame: Force coefficient dataframe
    """
    time_normalized = raw_cp["time_normalized"].copy()
    cols_points = [c for c in raw_cp.columns if c != "time_normalized"]
    id_points = np.array([int(c) for c in cols_points])

    points_selection = geometry_df.sort_values(by="point_idx")["point_idx"].to_numpy()
    face_area = geometry_df["area"].to_numpy()
    face_ns = geometry_df[["n_x", "n_y", "n_z"]].to_numpy().T

    mask_valid = np.isin(id_points, points_selection)
    id_points_selected = id_points[mask_valid]
    cp_matrix = raw_cp[cols_points].to_numpy()[:, mask_valid]

    regions_list = geometry_df["region_idx"].unique()

    f_matrix_x = -cp_matrix * face_area * face_ns[0, :]
    f_matrix_y = -cp_matrix * face_area * face_ns[1, :]
    f_matrix_z = -cp_matrix * face_area * face_ns[2, :]

    list_of_cf_region = []
    for region in regions_list:
        pts = geometry_df[geometry_df["region_idx"] == region]["point_idx"].to_numpy()
        mask = np.isin(id_points_selected, pts)
        list_of_cf_region.append(
            pd.DataFrame(
                {
                    "time_normalized": time_normalized,
                    "fx": np.sum(f_matrix_x[:, mask], axis=1),
                    "fy": np.sum(f_matrix_y[:, mask], axis=1),
                    "fz": np.sum(f_matrix_z[:, mask], axis=1),
                    "region_idx": region,
                }
            )
        )

    cf_full = pd.concat(list_of_cf_region)
    del list_of_cf_region

    Cf_data = (
        cf_full.groupby(["region_idx", "time_normalized"])
        .agg(
            Fx=pd.NamedAgg(column="fx", aggfunc="sum"),
            Fy=pd.NamedAgg(column="fy", aggfunc="sum"),
            Fz=pd.NamedAgg(column="fz", aggfunc="sum"),
        )
        .reset_index()
    )

    if nominal_area > 0:
        Cf_data["Cfx"] = Cf_data["Fx"] / nominal_area
        Cf_data["Cfy"] = Cf_data["Fy"] / nominal_area
        Cf_data["Cfz"] = Cf_data["Fz"] / nominal_area
        Cf_data.drop(columns=["Fx", "Fy", "Fz"], inplace=True)
    else:
        region_group_by = geometry_df.groupby(["region_idx"])
        rep_areas = {}
        for region_idx, region_points in region_group_by:
            pts_idx = region_points.point_idx.to_numpy()
            (Lx, Ly, Lz), (Ax, Ay, Az) = get_representative_areas(
                input_mesh=geometry, point_idx=pts_idx
            )
            rep_areas[region_idx[0]] = {
                "ATx": Ax,
                "ATy": Ay,
                "ATz": Az,
                "Lx": Lx,
                "Ly": Ly,
                "Lz": Lz,
            }
        rep_df = pd.DataFrame.from_dict(rep_areas, orient="index").reset_index()
        rep_df = rep_df.rename(columns={"index": "region_idx"})
        Cf_data = pd.merge(Cf_data, rep_df, on="region_idx")
        Cf_data["Cfx"] = Cf_data["Fx"] / Cf_data["ATx"]
        Cf_data["Cfy"] = Cf_data["Fy"] / Cf_data["ATy"]
        Cf_data["Cfz"] = Cf_data["Fz"] / Cf_data["ATz"]
        Cf_data.drop(columns=["Fx", "Fy", "Fz", "ATx", "ATy", "ATz"], inplace=True)

    return Cf_data


def process_Cf(
    mesh: LnasFormat,
    cfg: CfConfig,
    cp_h5: pathlib.Path,
    bodies_definition: dict[str, BodyDefinition],
    path_manager=None,
) -> dict[str, CommonOutput]:
    """Orchestrate Cf computation: tabulate geometry, run timeseries, stats.

    Args:
        mesh (LnasFormat): Input mesh
        cfg (CfConfig): Force coefficient configuration
        cp_h5 (pathlib.Path): Cp timeseries H5
        bodies_definition (dict[str, BodyDefinition]): Bodies definition dict
        path_manager: Optional path manager (unused here, for API compatibility)

    Returns:
        dict[str, CommonOutput]: Outputs keyed by direction
    """
    geometry_dict: dict[str, GeometryData] = {}
    for body_cfg in cfg.bodies:
        geom_data = get_geometry_data(
            body_cfg=body_cfg,
            sfc_list=bodies_definition[body_cfg.name].surfaces,
            mesh=mesh,
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

    def wrapper_transform_Cf(
        raw_cp: pd.DataFrame, geom_df: pd.DataFrame, geom: LnasGeometry
    ):
        return transform_Cf(raw_cp, geom_df, geom, nominal_area=cfg.nominal_area)

    Cf_data = process_timeseries(
        cp_h5=cp_h5,
        geometry_df=geometry_df,
        geometry=geometry_to_use,
        processing_function=wrapper_transform_Cf,
    )

    region_definition_df = get_region_definition_dataframe(geometry_dict)
    length_df = Cf_data[["region_idx"]].drop_duplicates()
    region_definition_df = pd.merge(
        region_definition_df, length_df, on="region_idx", how="left"
    )

    compiled_output = {}
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

        body_stats_by_tri: dict[str, pd.DataFrame] = {}
        for body_cfg in cfg.bodies:
            body_data = geometry_dict[body_cfg.name]
            region_idx_arr = geometry_df.loc[
                geometry_df.region_idx.str.contains(body_cfg.name)
            ].region_idx.to_numpy()
            body_stats_by_tri[body_cfg.name] = combine_stats_data_with_mesh(
                mesh=body_data.mesh,
                region_idx_array=region_idx_arr,
                data_stats=Cf_stats,
            )

        compiled_output[direction_lbl] = CommonOutput(
            data_df=Cf_dir_data,
            stats_df=Cf_stats,
            region_indexing_df=geometry_df[["region_idx", "point_idx"]],
            region_definition_df=region_definition_df,
        )

    return compiled_output


# ---------------------------------------------------------------------------
# Moment coefficient (Cm)
# ---------------------------------------------------------------------------


def add_lever_arm_to_geometry_df(
    geom_data: GeometryData,
    transformation,
    lever_origin: tuple[float, float, float],
    geometry_df: pd.DataFrame,
) -> pd.DataFrame:
    """Add lever arm distances to geometry_df for moment calculations.

    Args:
        geom_data (GeometryData): Geometry data object
        transformation: Transformation config
        lever_origin (tuple[float, float, float]): Lever origin coordinates
        geometry_df (pd.DataFrame): Dataframe with geometric properties

    Returns:
        pd.DataFrame: geometry_df merged with lever arm columns rx, ry, rz
    """
    transformed_body = geom_data.mesh.copy()
    transformed_body.apply_transformation(transformation.get_geometry_transformation())
    centroids = np.mean(transformed_body.triangle_vertices, axis=1)

    position_df = pd.DataFrame(
        {
            "rx": centroids[:, 0] - lever_origin[0],
            "ry": centroids[:, 1] - lever_origin[1],
            "rz": centroids[:, 2] - lever_origin[2],
            "point_idx": geom_data.triangles_idxs,
        }
    )
    return pd.merge(geometry_df, position_df, on="point_idx", how="left")


def get_representative_volume(
    input_mesh: LnasGeometry, point_idx: np.ndarray
) -> tuple[tuple[float, float, float], float]:
    """Calculate representative volume from the bounding box.

    Args:
        input_mesh (LnasGeometry): Input LNAS mesh
        point_idx (np.ndarray): Triangle indices

    Returns:
        tuple: ((Lx, Ly, Lz), V_rep)
    """
    geom_verts = input_mesh.triangle_vertices[point_idx].reshape(-1, 3)
    x_min, x_max = geom_verts[:, 0].min(), geom_verts[:, 0].max()
    y_min, y_max = geom_verts[:, 1].min(), geom_verts[:, 1].max()
    z_min, z_max = geom_verts[:, 2].min(), geom_verts[:, 2].max()

    Lx = max(x_max - x_min, 1.0)
    Ly = max(y_max - y_min, 1.0)
    Lz = max(z_max - z_min, 1.0)

    return (Lx, Ly, Lz), Lx * Ly * Lz


def transform_Cm(
    raw_cp: pd.DataFrame,
    geometry_df: pd.DataFrame,
    geometry: LnasGeometry,
    *,
    nominal_volume: float,
) -> pd.DataFrame:
    """Transform pressure coefficient into moment coefficient.

    Args:
        raw_cp (pd.DataFrame): Body Cp data (matrix form)
        geometry_df (pd.DataFrame): Geometric properties with lever arms
        geometry (LnasGeometry): Mesh geometry
        nominal_volume (float): Nominal volume for Cm calculation

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

    mask_valid = np.isin(id_points, points_selection)
    id_points_selected = id_points[mask_valid]
    cp_matrix = raw_cp[cols_points].to_numpy()[:, mask_valid]

    regions_list = geometry_df["region_idx"].unique()

    f_matrix_x = -cp_matrix * face_area * face_ns[0, :]
    f_matrix_y = -cp_matrix * face_area * face_ns[1, :]
    f_matrix_z = -cp_matrix * face_area * face_ns[2, :]

    m_matrix_x = face_pos[1, :] * f_matrix_z - face_pos[2, :] * f_matrix_y
    m_matrix_y = face_pos[2, :] * f_matrix_x - face_pos[0, :] * f_matrix_z
    m_matrix_z = face_pos[0, :] * f_matrix_y - face_pos[1, :] * f_matrix_x

    list_of_cm_region = []
    for region in regions_list:
        pts = geometry_df[geometry_df["region_idx"] == region]["point_idx"].to_numpy()
        mask = np.isin(id_points_selected, pts)
        list_of_cm_region.append(
            pd.DataFrame(
                {
                    "time_normalized": time_normalized,
                    "mx": np.sum(m_matrix_x[:, mask], axis=1),
                    "my": np.sum(m_matrix_y[:, mask], axis=1),
                    "mz": np.sum(m_matrix_z[:, mask], axis=1),
                    "region_idx": region,
                }
            )
        )

    cm_full = pd.concat(list_of_cm_region)
    del list_of_cm_region

    Cm_data = (
        cm_full.groupby(["region_idx", "time_normalized"])
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
        rep_volumes = {}
        for region_idx, region_points in region_group_by:
            pts_idx = region_points.point_idx.to_numpy()
            (Lx, Ly, Lz), V_rep = get_representative_volume(
                input_mesh=geometry, point_idx=pts_idx
            )
            rep_volumes[region_idx[0]] = {
                "V_rep": V_rep,
                "Lx": Lx,
                "Ly": Ly,
                "Lz": Lz,
            }
        rep_df = pd.DataFrame.from_dict(rep_volumes, orient="index").reset_index()
        rep_df = rep_df.rename(columns={"index": "region_idx"})
        Cm_data = pd.merge(Cm_data, rep_df, on="region_idx")
        Cm_data["Cmx"] = Cm_data["Mx"] / Cm_data["V_rep"]
        Cm_data["Cmy"] = Cm_data["My"] / Cm_data["V_rep"]
        Cm_data["Cmz"] = Cm_data["Mz"] / Cm_data["V_rep"]
        Cm_data.drop(columns=["Mx", "My", "Mz", "V_rep"], inplace=True)

    return Cm_data


def process_Cm(
    mesh: LnasFormat,
    cfg: CmConfig,
    cp_h5: pathlib.Path,
    bodies_definition: dict[str, BodyDefinition],
    path_manager=None,
) -> dict[str, CommonOutput]:
    """Orchestrate Cm computation: tabulate geometry, run timeseries, stats.

    Args:
        mesh (LnasFormat): Input mesh
        cfg (CmConfig): Moment coefficient configuration
        cp_h5 (pathlib.Path): Cp timeseries H5
        bodies_definition (dict[str, BodyDefinition]): Bodies definition dict
        path_manager: Optional path manager (unused here, for API compatibility)

    Returns:
        dict[str, CommonOutput]: Outputs keyed by direction
    """
    geometry_dict: dict[str, GeometryData] = {}
    for body_cfg in cfg.bodies:
        geom_data = get_geometry_data(
            body_cfg=body_cfg,
            sfc_list=bodies_definition[body_cfg.name].surfaces,
            mesh=mesh,
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
        raw_cp: pd.DataFrame, geom_df: pd.DataFrame, geom: LnasGeometry
    ):
        return transform_Cm(raw_cp, geom_df, geom, nominal_volume=cfg.nominal_volume)

    Cm_data = process_timeseries(
        cp_h5=cp_h5,
        geometry_df=geometry_df,
        geometry=geometry_to_use,
        processing_function=wrapper_transform_Cm,
    )

    region_definition_df = get_region_definition_dataframe(geometry_dict)
    length_df = Cm_data[["region_idx"]].drop_duplicates()
    region_definition_df = pd.merge(
        region_definition_df, length_df, on="region_idx", how="left"
    )

    compiled_output = {}
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
        compiled_output[direction_lbl] = CommonOutput(
            data_df=Cm_dir_data,
            stats_df=Cm_stats,
            region_indexing_df=geometry_df[["region_idx", "point_idx"]],
            region_definition_df=region_definition_df,
        )

    return compiled_output


# ---------------------------------------------------------------------------
# Shape coefficient (Ce)
# ---------------------------------------------------------------------------


def transform_Ce(
    raw_cp: pd.DataFrame,
    geometry_df: pd.DataFrame,
    _geometry: LnasGeometry,
) -> pd.DataFrame:
    """Transform pressure coefficient into shape coefficient.

    Args:
        raw_cp (pd.DataFrame): Body Cp data (matrix form)
        geometry_df (pd.DataFrame): Geometric properties and triangle indexing
        _geometry (LnasGeometry): Unused (kept for consistent function signature)

    Returns:
        pd.DataFrame: Shape coefficient dataframe
    """
    time_normalized = raw_cp["time_normalized"].copy()
    cols_points = [c for c in raw_cp.columns if c != "time_normalized"]
    id_points = np.array([int(c) for c in cols_points])

    points_selection = geometry_df.sort_values(by="point_idx")["point_idx"].to_numpy()
    face_area = geometry_df["area"].to_numpy()

    mask_valid = np.isin(id_points, points_selection)
    id_points_selected = id_points[mask_valid]
    cp_matrix = raw_cp[cols_points].to_numpy()[:, mask_valid]

    regions_list = geometry_df["region_idx"].unique()
    f_q_matrix = cp_matrix * face_area

    list_of_ce_region = []
    for region in regions_list:
        pts = geometry_df[geometry_df["region_idx"] == region]["point_idx"].to_numpy()
        mask = np.isin(id_points_selected, pts)
        list_of_ce_region.append(
            pd.DataFrame(
                {
                    "time_normalized": time_normalized,
                    "f/q": np.sum(f_q_matrix[:, mask], axis=1),
                    "area": np.sum(face_area[mask]),
                    "region_idx": region,
                }
            )
        )

    ce_full = pd.concat(list_of_ce_region)
    del list_of_ce_region

    Ce_data = (
        ce_full.groupby(["region_idx", "time_normalized"])
        .agg(
            total_area=pd.NamedAgg(column="area", aggfunc="sum"),
            total_force=pd.NamedAgg(column="f/q", aggfunc="sum"),
        )
        .reset_index()
    )

    Ce_data["Ce"] = Ce_data["total_force"] / Ce_data["total_area"]
    Ce_data.drop(columns=["total_area", "total_force"], inplace=True)

    return Ce_data


def get_surface_dict(cfg: CeConfig, mesh: LnasFormat) -> dict[str, list[str]]:
    """Generate a dictionary with surface names keyed by surface or set name.

    Args:
        cfg (CeConfig): Shape coefficient configuration
        mesh (LnasFormat): Input mesh

    Returns:
        dict[str, list[str]]: Surface definition dictionary
    """
    sfc_dict = {set_lbl: sfc_list for set_lbl, sfc_list in cfg.sets.items()}
    sfc_dict |= {sfc: [sfc] for sfc in mesh.surfaces if sfc not in cfg.surfaces_in_sets}
    return sfc_dict


def process_surfaces(
    geometry_dict: dict[str, GeometryData], cfg: CeConfig, ce_stats: pd.DataFrame
) -> tuple[list[ProcessedEntity], pd.DataFrame]:
    """Generate a ProcessedEntity for each body surface.

    Args:
        geometry_dict (dict[str, GeometryData]): Geometry data keyed by surface label
        cfg (CeConfig): Shape coefficient configuration
        ce_stats (pd.DataFrame): Statistical values per region per surface

    Returns:
        tuple: (list of ProcessedEntity, region indexing DataFrame)
    """
    processed_surfaces: list[ProcessedEntity] = []
    region_indexing_dfs = []
    last_index = 0

    for sfc_lbl, geom_data in geometry_dict.items():
        regions_mesh, indexing = generate_regions_mesh(geom_data=geom_data, cfg=cfg)
        indexing = np.char.add(indexing.astype(str), "-" + sfc_lbl)

        if (
            pd.merge(
                pd.DataFrame({"region_idx": np.unique(indexing)}),
                ce_stats.reset_index(drop=False),
                on="region_idx",
                how="left",
            )
            .isnull()
            .sum()
            .any()
        ):
            logger.warning(
                "Region refinement is greater than data refinement. Resulted in NaN values"
            )

        region_indexing_dfs.append(
            pd.DataFrame(
                {
                    "point_idx": np.arange(len(regions_mesh.triangle_vertices)) + last_index,
                    "region_idx": indexing,
                }
            )
        )
        last_index += len(regions_mesh.triangle_vertices)
        processed_surfaces.append(ProcessedEntity(mesh=regions_mesh))

    return processed_surfaces, pd.concat(region_indexing_dfs)


def process_Ce(
    mesh: LnasFormat,
    cfg: CeConfig,
    cp_h5: pathlib.Path,
    path_manager=None,
) -> CeOutput:
    """Orchestrate Ce computation: geometry, timeseries, stats, region meshes.

    Args:
        mesh (LnasFormat): Input mesh
        cfg (CeConfig): Shape coefficient configuration
        cp_h5 (pathlib.Path): Cp timeseries H5
        path_manager: Optional path manager (unused, for API compatibility)

    Returns:
        CeOutput: Compiled outputs
    """
    mesh_areas = mesh.geometry.areas
    mesh_normals = mesh.geometry.normals

    sfc_dict = get_surface_dict(cfg=cfg, mesh=mesh)

    logger.info("Getting geometry data...")
    geometry_dict = get_ce_geometry_data(surface_dict=sfc_dict, cfg=cfg, mesh=mesh)

    logger.info("Tabulating geometry data...")
    geometry_df = tabulate_geometry_data(
        geom_dict=geometry_dict,
        mesh_areas=mesh_areas,
        mesh_normals=mesh_normals,
        transformation=cfg.transformation,
    )

    logger.info("Processing timeseries...")
    Ce_data_raw = process_timeseries(
        cp_h5=cp_h5,
        geometry_df=geometry_df,
        geometry=mesh.geometry,
        processing_function=transform_Ce,
    )
    Ce_data = convert_dataframe_into_matrix(
        Ce_data_raw,
        row_data_label="time_normalized",
        column_data_label="region_idx",
        value_data_label="Ce",
    )

    logger.info("Calculating statistics...")
    Ce_stats = calculate_statistics(Ce_data, statistics_to_apply=cfg.statistics)

    logger.info("Processing surfaces...")
    processed_surfaces, regions_indexing_df = process_surfaces(
        geometry_dict=geometry_dict, cfg=cfg, ce_stats=Ce_stats
    )

    excluded_sfc_list = [sfc for sfc in cfg.zoning.exclude if sfc in mesh.surfaces]  # type: ignore
    excluded_sfc_list += [
        sfc
        for set_lbl, sfc_set in cfg.sets.items()
        for sfc in sfc_set
        if set_lbl in cfg.zoning.exclude  # type: ignore
    ]
    if excluded_sfc_list:
        excluded_sfcs, _ = mesh.geometry_from_list_surfaces(
            surfaces_names=excluded_sfc_list
        )
        excluded_entities = [ProcessedEntity(mesh=excluded_sfcs)]
    else:
        excluded_entities = []

    return CeOutput(
        processed_entities=processed_surfaces,
        excluded_entities=excluded_entities,
        data_df=Ce_data,
        stats_df=Ce_stats,
        region_indexing_df=regions_indexing_df[["region_idx", "point_idx"]],
        region_definition_df=get_region_definition_dataframe(geometry_dict),
    )
