from __future__ import annotations

import pathlib
from typing import Literal

import numpy as np
import pandas as pd
from pydantic import BaseModel, ConfigDict

from cfdmod.use_cases.hfpi import common


class DimensionalData(BaseModel):
    """Analytical data required to analyze a given HFPI model"""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    U_H: float
    height: float
    base: float

    @property
    def dynamic_pressure(self):
        return 0.613 * (self.U_H**2)

    @property
    def CST(self):
        return min(self.base, self.height) / self.U_H

    @property
    def time_normalization_factor(self):
        return self.CST

    @property
    def force_normalization_factor(self):
        return self.base * self.height * self.dynamic_pressure

    @property
    def moments_normalization_factor(self):
        return self.base * self.base * self.height * self.dynamic_pressure


def read_static_forces(hdf_path: pathlib.Path) -> pd.DataFrame:
    """Read forces for HFPI from path, with scalar key specified"""
    df_force = pd.read_hdf(hdf_path)
    req_keys = ["time_normalized"]
    if not common.validate_keys_df(df_force, req_keys):
        raise KeyError(
            f"Not all required keys ({req_keys}) present in HFPI Forces HDF {hdf_path.as_posix()}. Found only keys {df_force.columns}"
        )

    # Remove points that are not in any area (-1 label)
    col_neg = next(iter(k for k in df_force.columns if isinstance(k, str) and k.startswith('-1')), None)
    if(col_neg is not None):
        df_force.drop(columns=[col_neg], inplace=True)

    return df_force

def remove_suffix_static_forces(df_forces: pd.DataFrame):
    """Remove the suffixes for static forces in df_forces"""
    columns = list(df_forces.columns)
    rename_colums = {k: k.split("-")[0] for k in columns if "-" in k and k != "time_normalized"}
    df_forces.rename(columns=rename_colums, inplace=True)


def scale_forces(
    df_force: pd.DataFrame, key_name: str, *, force_factor: float, time_factor: float
):
    """Normalize HFPI forces by given factors"""

    df_force["time"] = df_force["time_normalized"] * time_factor
    df_force.drop("time_normalized", axis=1, inplace=True)
    col_mul = [k for k in df_force.columns if not isinstance(k, str) or not k.startswith("time")]
    df_force[col_mul] *= force_factor
    df_force.sort_values(by=["time"], inplace=True)


class StaticForcesData(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)

    cf_x: pd.DataFrame
    cf_y: pd.DataFrame
    cm_z: pd.DataFrame
    is_scaled: bool = False

    def get_as_dct(self):
        col_order = sorted([c for c in self.cf_x.columns if c.isnumeric()], key=lambda v: int(v))

        def df2np(df: pd.DataFrame):
            return df[col_order].to_numpy()

        return {"x": df2np(self.cf_x), "y": df2np(self.cf_y), "z": df2np(self.cm_z)}

    def fill_missing_floors(self, n_floors: int):
        for df in [self.cf_x, self.cf_y, self.cm_z]:
            common.fill_forces_floors(df, n_floors)

    @property
    def n_samples(self):
        return len(self.cf_x)

    @property
    def delta_t(self):
        k = "time_normalized"
        if "time" in self.cf_x.columns:
            k = "time"
        return self.cf_x[k][1] - self.cf_x[k][0]

    @classmethod
    def build(cls, cf_x_h5: pathlib.Path, cf_y_h5: pathlib.Path, cm_z_h5: pathlib.Path, remove_suffix: bool = False):
        cf_x = read_static_forces(cf_x_h5)
        cf_y = read_static_forces(cf_y_h5)
        cm_z = read_static_forces(cm_z_h5)
        if len(cf_x) != len(cf_y) or len(cf_x) != len(cm_z):
            raise ValueError(
                f"Length of forces data don't match. Paths {cf_x_h5, cf_y_h5, cm_z_h5}"
            )

        if(remove_suffix):
            for df in (cf_x, cf_y, cm_z):
                remove_suffix_static_forces(df)

        return StaticForcesData(
            cf_x=cf_x,
            cf_y=cf_y,
            cm_z=cm_z,
        )

    def get_scaled_forces(self, dim_data: DimensionalData) -> StaticForcesData:
        """Generate HFPI scaled forces data"""
        time_factor = dim_data.time_normalization_factor

        if self.is_scaled:
            raise ValueError("Forces were already scaled, unable to scale again")

        cf_x = self.cf_x.copy()
        cf_y = self.cf_y.copy()
        cm_z = self.cm_z.copy()

        scale_forces(
            cf_x,
            "FX",
            force_factor=dim_data.force_normalization_factor,
            time_factor=time_factor,
        )
        scale_forces(
            cf_y,
            "FY",
            force_factor=dim_data.force_normalization_factor,
            time_factor=time_factor,
        )
        scale_forces(
            cm_z,
            "MZ",
            force_factor=dim_data.moments_normalization_factor,
            time_factor=time_factor,
        )

        return StaticForcesData(
            cf_x=cf_x,
            cf_y=cf_y,
            cm_z=cm_z,
            is_scaled=True,
        )


class StaticResults(BaseModel):
    """Results generated from static analysis"""

    model_config = ConfigDict(arbitrary_types_allowed=True)
    floors_heights: np.ndarray

    # data as ["x", "y", "z"] = time series
    forces_static: dict[str, np.ndarray]
    moments_static: dict[str, np.ndarray]

    @property
    def global_forces_static(self):
        return common.get_global_dct(self.forces_static)

    @property
    def global_moments_static(self):
        return common.get_global_dct(self.moments_static)

    def get_stats_forces_static(self, stats_type: Literal["min", "max", "mean"]):
        return common.get_stats_dct(self.forces_static, stats_type)

    def get_stats_moments_static(self, stats_type: Literal["min", "max", "mean"]):
        return common.get_stats_dct(self.moments_static, stats_type)

    def get_stats_global_forces_static(self, stats_type: Literal["min", "max", "mean"]):
        return common.get_stats_dct(self.global_forces_static, stats_type)

    def get_stats_global_moments_static(self, stats_type: Literal["min", "max", "mean"]):
        return common.get_stats_dct(self.global_moments_static, stats_type)


def validate_forces_w_n_floors(forces: StaticForcesData, n_floors: int):
    for name, df in [("Cfx", forces.cf_x), ("Cfy", forces.cf_y), ("Cmz", forces.cm_z)]:
        cols = df.columns
        for k in range(n_floors):
            if int(k) not in cols and str(k) not in cols:
                raise KeyError(
                    f"Coefficient {name} doesn't have all floors available. Nº of floors: {n_floors}; Columns found: {cols}"
                )


def solve_static_forces(
    forces: StaticForcesData, dim_data: DimensionalData, floors_heights: np.ndarray
):
    """Solve system for static forces"""
    forces.fill_missing_floors(len(floors_heights))
    validate_forces_w_n_floors(forces, len(floors_heights))
    normalized_forces = forces.get_scaled_forces(dim_data)

    force_static = normalized_forces.get_as_dct()
    moments_static = common.get_moments_from_force(force_static, floors_heights)
    return StaticResults(
        floors_heights=floors_heights, forces_static=force_static, moments_static=moments_static
    )
